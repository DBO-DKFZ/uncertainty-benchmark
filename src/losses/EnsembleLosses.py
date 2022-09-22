from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# For abstract inheritance
from abc import abstractmethod

from ..utils.einop import eincheck, einop
from einops import rearrange, reduce, repeat, parse_shape

log_offset = 1e-10
root_offset = 1e-10
det_offset = 1e-6

LOG_OFFSET = 1e-10
ROOT_OFFSET = 1e-10
DET_OFFSET = 1e-6


def apply_loss_heads(outs, loss, target):
    indi_l = []

    if type(outs) == list:
        outs = torch.stack(outs, 0)

    num_ens = outs.size(0)
    for i in range(num_ens):
        tmp = loss(outs[i], target)
        indi_l.append(tmp)

    return indi_l


def pairwise_weightCos(weights_in, self_ortho: float, inter_ortho: float):

    num_ensembles = len(weights_in)
    num_layers = len(weights_in[0])

    for i in range(num_ensembles):
        assert len(weights_in[i]) == num_layers

    self_loss = 0.0
    inter_loss = 0.0

    def _compute_ortho_loss(kernel_1: torch.Tensor, kernel_2: torch.Tensor):

        assert kernel_1.shape == kernel_2.shape

        kernel_1 = rearrange(kernel_1, "in_c out_c h w -> out_c (h w in_c) ")
        kernel_2 = rearrange(kernel_2, "in_c out_c h w -> out_c (h w in_c) ")

        kernel_1 = nn.functional.normalize(kernel_1, 2, dim=-1)
        kernel_2 = nn.functional.normalize(kernel_2, 2, dim=-1)

        sim_matrix = torch.mm(kernel_1, kernel_2.t())
        mask = 1 - torch.eye(*sim_matrix.shape, device=sim_matrix.device)
        sim_matrix *= mask

        error = torch.sum(sim_matrix**2)
        return error

    for layer_idx in range(num_layers):

        for head_1_idx in range(num_ensembles):
            for head_2_idx in range(num_ensembles):

                if head_2_idx < head_1_idx:
                    continue

                if head_1_idx == head_2_idx and self_ortho:
                    self_loss += _compute_ortho_loss(
                        weights_in[head_1_idx][layer_idx],
                        weights_in[head_2_idx][layer_idx],
                    )

                if head_1_idx < head_2_idx and inter_ortho:
                    inter_loss += _compute_ortho_loss(
                        weights_in[head_1_idx][layer_idx],
                        weights_in[head_2_idx][layer_idx],
                    )

    return self_ortho * self_loss + inter_ortho * inter_loss


def pairwise_acti_diversity(activations: torch.Tensor, ens_div_para: float, mode: str):

    num_ensembles = len(activations)
    loss = 0.0

    def _compute_ortho_loss(kernel_1: torch.Tensor, kernel_2: torch.Tensor):

        assert kernel_1.shape == kernel_2.shape

        kernel_1 = rearrange(kernel_1, "b c h w -> b (h w) c")
        kernel_2 = rearrange(kernel_2, "b c h w -> b (h w) c")

        kernel_1 = nn.functional.normalize(kernel_1, 2, dim=-1)
        kernel_2 = nn.functional.normalize(kernel_2, 2, dim=-1)

        # Inner product
        dot_matrix = kernel_1 * kernel_2
        cos_sim = reduce(dot_matrix, "b i c -> b i", reduction="sum")

        # Positive
        cos_sim = cos_sim**2

        # Mean error over batch and image_dims
        error = reduce(cos_sim, "b i -> ", reduction="mean")

        return error

        # Batched matrix multiply
        # sim_matrix = torch.bmm(rearrange(kernel_1, "b i j -> b j i"), kernel_2)

        # Subtract diagonal elements
        # mask = 1 - torch.eye(*sim_matrix.shape[1:], device=sim_matrix.device)
        # mask = repeat(mask, "... -> batch_size ...", batch_size=sim_matrix.shape[0])
        # sim_matrix *= mask

        # Non-negative error
        # sim_matrix = sim_matrix**2

        # Take the mean over the matrices reduced by the zero-diagonal
        # sim_matrix = reduce(sim_matrix, "b i j -> b", reduction="sum")
        # sim_matrix /= kernel_1.size(1) * kernel_2.size(1) - kernel_1.size(1)

    def _compute_C2_loss(kernel_1: torch.Tensor, kernel_2: torch.Tensor):
        assert kernel_1.shape == kernel_2.shape

        kernel_1 = rearrange(kernel_1, "b c h w -> b (h w) c")
        kernel_2 = rearrange(kernel_2, "b c h w -> b (h w) c")

        kernel_1 = nn.functional.softmax(kernel_1, dim=-1)
        kernel_2 = nn.functional.softmax(kernel_2, dim=-1)

        c2_dist = 1 / 2 * torch.sum((kernel_1 - kernel_2) ** 2 / kernel_1 + kernel_2, -1)
        error = reduce(c2_dist, "b i -> ", reduction="mean")

        return error

    def _compute_KL_loss(kernel_1: torch.Tensor, kernel_2: torch.Tensor):
        """Computes the symmetric KL divergence 1/2KL(P||Q)+1/2KL(Q||P) also called the jeffrey-divergence."""

        assert kernel_1.shape == kernel_2.shape

        kernel_1 = rearrange(kernel_1, "b c h w -> b (h w) c")
        kernel_2 = rearrange(kernel_2, "b c h w -> b (h w) c")

        kernel_1 = nn.functional.softmax(kernel_1, dim=-1)
        kernel_2 = nn.functional.softmax(kernel_2, dim=-1)

        log_kernel_1 = torch.log(kernel_1)
        log_kernel_2 = torch.log(kernel_2)

        H_1 = torch.sum(kernel_1 * log_kernel_1, dim=-1)
        H_2 = torch.sum(kernel_2 * log_kernel_2, dim=-1)

        CH_12 = torch.sum(kernel_1 * log_kernel_2, dim=-1)
        CH_21 = torch.sum(kernel_2 * log_kernel_1, dim=-1)

        KL_12 = H_1 - CH_12
        KL_21 = H_2 - CH_21

        jeffrey_div = 1 / 2 * (KL_12 + KL_21)

        error = reduce(jeffrey_div, "b i -> ", reduction="mean")

        return error

    if mode == "cos":
        sim_func = _compute_ortho_loss
    elif mode == "chi2":
        sim_func = _compute_C2_loss
    elif mode == "KL":
        sim_func = _compute_KL_loss
    else:
        raise RuntimeError(f"Unknown similarity function {mode}")

    for head_1_idx in range(num_ensembles):
        for head_2_idx in range(num_ensembles):

            if head_2_idx <= head_1_idx:
                continue

            if head_1_idx < head_2_idx:
                loss += sim_func(
                    activations[head_1_idx],
                    activations[head_2_idx],
                )

    return ens_div_para * loss


class BaseMultiHeadLoss(nn.Module):
    def __init__(self, tradeoff, twoD=False, ignore_index=255):
        super(BaseMultiHeadLoss, self).__init__()

        if tradeoff > 1 or tradeoff < 0:
            print("Tradeoff should be in 0 to 1")
            exit()

        self.twoD = twoD
        self.tradeoff = tradeoff
        self.ignore_index = ignore_index

    def transform_input(self, outs, ens_out, mask):
        return outs, ens_out, mask

    @abstractmethod
    def compute_mean_loss(self, outs, ens_out, mask):
        return 0

    @abstractmethod
    def compute_indi_loss(self, outs, ens_out, mask):
        return 0, []

    @abstractmethod
    def compute_regularizer(self, outs, ens_out, mask):
        return 0

    def __call__(self, outs, ens_out, mask):

        outs, ens_out, mask = self.transform_input(outs, ens_out, mask)

        ens_loss = 0.0
        if self.tradeoff != 0.0:
            ens_loss = self.compute_mean_loss(outs, ens_out, mask)

        combined_loss = 0.0
        indi_loss = []
        if self.tradeoff != 1.0:
            combined_loss, indi_loss = self.compute_indi_loss(outs, ens_out, mask)

        reg = self.compute_regularizer(outs, ens_out, mask)

        return self.tradeoff * ens_loss + (1 - self.tradeoff) * combined_loss + reg


class MultiHeadCrossEntropyLoss(BaseMultiHeadLoss):
    def __init__(self, tradeoff, twoD=False, ignore_index=255):
        super(MultiHeadCrossEntropyLoss, self).__init__(tradeoff, twoD, ignore_index)

        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.NLL = nn.NLLLoss(ignore_index=ignore_index)

    def compute_mean_loss(self, outs, ens_out, mask):
        return self.NLL(torch.log(ens_out + LOG_OFFSET), mask)

    def compute_indi_loss(self, outs, ens_out, mask):
        indi_loss = apply_loss_heads(outs, self.CE, mask)
        return torch.sum(torch.stack(indi_loss)), indi_loss


class MulHCELossOrthoActi(MultiHeadCrossEntropyLoss):
    def __init__(
        self,
        tradeoff,
        mode: str,
        ens_div_para: float,
        acti_func=None,
        twoD=False,
        ignore_index=255,
    ):

        assert mode in ["cos", "KL", "chi2"]

        super(MulHCELossOrthoActi, self).__init__(tradeoff, twoD, ignore_index)

        self.acti_func = acti_func
        self.ens_div_para = ens_div_para
        self.mode = mode

        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.NLL = nn.NLLLoss(ignore_index=ignore_index)

    def compute_regularizer(self, outs, ens_out, mask):
        ens_div = pairwise_acti_diversity(self.acti_func(), self.ens_div_para, mode=self.mode)
        return ens_div


class MulHCELossKernelWeightCos(MultiHeadCrossEntropyLoss):
    def __init__(
        self,
        tradeoff,
        self_ortho=0.1,
        inter_ortho=0.1,
        weight_func=None,
        twoD=False,
        ignore_index=255,
    ):

        super(MulHCELossKernelWeightCos, self).__init__(tradeoff, twoD, ignore_index)

        self.weight_func = weight_func

        self.self_ortho = self_ortho
        self.inter_ortho = inter_ortho

        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.NLL = nn.NLLLoss(ignore_index=ignore_index)

    def compute_regularizer(self, outs, ens_out, mask):
        ens_div = pairwise_weightCos(self.weight_func(), self_ortho=self.self_ortho, inter_ortho=self.inter_ortho)
        return ens_div

    # def __call__(self, outs, ens_out, mask):

    # head_loss, hist = apply_loss_heads(outs, self.CE, mask)
    # ens_loss = self.NLL(torch.log(ens_out + log_offset), mask)

    # return self.tradeoff * ens_loss + (1 - self.tradeoff) * head_loss - ens_div


if __name__ == "__main__":
    from ..pl_modules.models.multiHeadResnet import multiHead_resnet18

    a = torch.rand([5, 3, 20, 20])
    net = multiHead_resnet18(2, 0, 2, None)
    list_out = net(a)

    loss = MulHCELossKernelWeightCos(
        0.0,
        1.0,
        weight_func=lambda: net.getEnsembleParams(filter_types=(torch.nn.Conv2d,)),
        twoD=False,
    )

    ens_mean = reduce(torch.stack(list_out, dim=0), "num_ens ... -> ...", reduction="mean")

    l = loss(list_out, ens_mean, torch.zeros(5, dtype=torch.long))
