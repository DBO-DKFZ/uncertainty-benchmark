import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

import torchvision.transforms as T
from ..datamodules.camelyon_datamodules import DiscreteRotationTransform
from ..datamodules.datasets.camelyon import Camelyon17

from ..losses import EnsembleLosses
import argparse

from .basemodules import (
    LitTileClassifier,
    EnsembleTileClassifier,
    log_test_metrics,
    log_train_metrics,
    log_val_metrics,
    display_batch,
)
from .models.multiHeadResnet import *
from .models.dropoutResnet import *
from .models.SVIResNet import *

DEBUG = False


@MODEL_REGISTRY
class ResNetClassifier(LitTileClassifier):

    res_map = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(
        self,
        size: int,
        num_classes: int,
        lr: float = 1e-4,
        opti_metric: str = "val_loss",
        patience: int = 3,
    ):

        assert size in ResNetClassifier.res_map

        net = ResNetClassifier.res_map[size](num_classes=num_classes)

        super().__init__(net, num_classes, lr=lr)


@MODEL_REGISTRY
class TTAResNetClassifier(ResNetClassifier):
    def __init__(
        self,
        size: int,
        num_classes: int,
        lr: float = 1e-4,
        opti_metric: str = "val_loss",
        patience: int = 3,
        tta_iterations: int = 10,
    ):

        super().__init__(size=size, num_classes=num_classes, lr=lr, opti_metric=opti_metric, patience=patience)

        self.transforms = None
        self.unnorm = T.Compose(
            [
                T.Normalize(0, 1 / Camelyon17.MEAN_STD[1]),
                T.Normalize(-Camelyon17.MEAN_STD[0], 1),
            ]
        )
        self.tta_iterations = tta_iterations

    def set_transforms(self, augmentations: str = "crop"):
        # base_transforms = [T.ToTensor(), T.Normalize(*Camelyon17.MEAN_STD)]
        if augmentations == "crop":
            tta_transforms = []
        elif augmentations == "flip":
            tta_transforms = [
                # Unnormalize before applying transforms
                T.Normalize(0, 1 / Camelyon17.MEAN_STD[1]),
                T.Normalize(-Camelyon17.MEAN_STD[0], 1),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                DiscreteRotationTransform(angles=[-90, 0, 90]),
                T.Normalize(*Camelyon17.MEAN_STD),
            ]
        elif augmentations == "strong":
            tta_transforms = [
                # Unnormalize before applying transforms
                T.Normalize(0, 1 / Camelyon17.MEAN_STD[1]),
                T.Normalize(-Camelyon17.MEAN_STD[0], 1),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                DiscreteRotationTransform(angles=[-90, 0, 90]),
                T.RandomAffine(degrees=0, scale=(1.0, 1.2)),
                T.GaussianBlur(kernel_size=(5, 5), sigma=(1e-9, 0.1)),
                T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.1),
                T.Normalize(*Camelyon17.MEAN_STD),
            ]
        else:
            raise RuntimeError("Augmentations must be in [crop, flip, strong]")
        self.transforms = T.Compose(tta_transforms)

    def test_step(self, batch, batch_idx: int, d_idx: int = 0):

        assert self.transforms is not None, "TTA Transforms not set"
        # loss, out_sm, preds, y = self.evaluate(batch)
        img, y = batch
        # tta_imgs = img.permute(1, 0, 2, 3, 4) # Only needed for albumentation transforms
        if DEBUG:
            # display_batch(batch)
            tta_out = []
            imgs = []
            for i in range(self.tta_iterations):
                tta_img = self.transforms(img)
                tta_out.append(self.forward(tta_img))
                unnorm_img = self.unnorm(tta_img)
                np_img = unnorm_img.detach().cpu().numpy()
                np_img = np_img.transpose(0, 2, 3, 1)
                np_img = np.array(np_img, dtype=np.float32)
                imgs.append(np_img[0])
            fig, axs = plt.subplots(4, 5)
            axs = axs.flatten()
            for idx in range(len(imgs)):
                img = imgs[idx]
                # Unnormalize image
                # img = (img - np.min(img)) / (np.max(img) - np.min(img))
                axs[idx].imshow(img)
                axs[idx].axis("off")
            plt.show()
        else:
            tta_out = []
            for i in range(self.tta_iterations):
                tta_img = self.transforms(img)
                tta_out.append(self.forward(tta_img))
        tta_out = torch.stack(tta_out)
        # tta_out has shape (tta_iterations, batch_size, num_classes)
        tta_sm = F.softmax(tta_out, dim=2)
        var_sm, mean_sm = torch.var_mean(tta_sm, dim=0)
        conf, preds = torch.max(mean_sm, dim=1)

        loss = F.cross_entropy(tta_out.mean(dim=0), y)

        log_test_metrics(self, d_idx, preds, y, mean_sm, loss)
        return {"d_idx": d_idx, "softmax": mean_sm, "label": y, "variance": var_sm}

    def predict_step(self, batch, batch_idx: int, d_idx: int = 0):

        # loss, out_sm, preds, y = self.evaluate(batch)
        img, y = batch

        tta_out = []
        for i in range(self.tta_iterations):
            tta_img = self.transforms(img)
            tta_out.append(self.forward(tta_img))
        tta_out = torch.stack(tta_out)
        # tta_out has shape (tta_iterations, batch_size, num_classes)
        tta_sm = F.softmax(tta_out, dim=2)
        var_sm, mean_sm = torch.var_mean(tta_sm, dim=0)

        return {"softmax": mean_sm, "label": y, "variance": var_sm}


@MODEL_REGISTRY
class MCDOResNetClassifier(LitTileClassifier):

    res_map = {
        18: mcdo_resnet18,
        34: mcdo_resnet34,
        50: mcdo_resnet50,
        101: mcdo_resnet101,
        152: mcdo_resnet152,
    }

    def __init__(
        self,
        size: int,
        num_classes: int,
        mc_iterations: int = 20,
        dropout: float = 0.4,
        lr: float = 1e-4,
        opti_metric: str = "val_loss",
        patience: int = 3,
    ):

        assert size in MCDOResNetClassifier.res_map

        net = MCDOResNetClassifier.res_map[size](num_classes, dropout)
        super().__init__(net, num_classes, lr=lr)
        self.mc_iterations = mc_iterations

    # Overwrite test-loop to perform multiple predictions
    def test_step(self, batch, batch_idx: int, d_idx: int = 0):
        # Enable Monte Carlo Dropout
        self.model.dropout.train()
        x, y = batch
        mc_out = torch.vstack([self.model(x).unsqueeze(0) for _ in range(self.mc_iterations)])
        # mc_out has shape (mc_iterations, batch_size, num_classes)
        mc_sm = F.softmax(mc_out, dim=2)
        var_sm, mean_sm = torch.var_mean(mc_sm, dim=0)
        conf, preds = torch.max(mean_sm, dim=1)

        loss = F.cross_entropy(mc_out.mean(dim=0), y)
        log_test_metrics(self, d_idx, preds, y, mean_sm, loss)
        return {"d_idx": d_idx, "softmax": mean_sm, "label": y, "variance": var_sm}

    # Overwrite test-loop to perform multiple predictions
    def predict_step(self, batch, batch_idx: int, d_idx: int = 0):
        # Enable Monte Carlo Dropout
        self.model.dropout.train()
        x, y = batch
        mc_out = torch.vstack([self.model(x).unsqueeze(0) for _ in range(self.mc_iterations)])
        # mc_out has shape (mc_iterations, batch_size, num_classes)
        mc_sm = F.softmax(mc_out, dim=2)
        var_sm, mean_sm = torch.var_mean(mc_sm, dim=0)
        return {"softmax": mean_sm, "label": y, "variance": var_sm}


@MODEL_REGISTRY
class SVIResNetClassifier(LitTileClassifier):

    res_map = {
        18: svi_resnet18,
        34: svi_resnet34,
        50: svi_resnet50,
        101: svi_resnet101,
        152: svi_resnet152,
    }

    def __init__(
        self,
        size: int,
        num_classes: int,
        svi_samples: int = 20,
        lr: float = 1e-4,
        opti_metric: str = "val_loss",
        patience: int = 3,
        inference: str = "ffg",
        kl_weight: float = 1.0,
        svi_train_samples: int = 1,
        svi_val_samples: int = 1,
    ):

        assert size in SVIResNetClassifier.res_map

        net = SVIResNetClassifier.res_map[size](num_classes, inference)
        super().__init__(net, num_classes, lr=lr)
        self.kl_weight = kl_weight
        self.svi_samples = svi_samples
        self.svi_train_samples = svi_train_samples
        self.svi_val_samples = svi_val_samples

    # Overwrite test-loop to perform multiple predictions
    def test_step(self, batch, batch_idx: int, d_idx: int = 0):
        x, y = batch
        svi_out = torch.vstack([self.model(x).unsqueeze(0) for _ in range(self.svi_samples)])
        # mc_out has shape (mc_iterations, batch_size, num_classes)
        svi_sm = F.softmax(svi_out, dim=2)
        var_sm, mean_sm = torch.var_mean(svi_sm, dim=0)
        conf, preds = torch.max(mean_sm, dim=1)

        loss = F.cross_entropy(svi_out.mean(dim=0), y)
        kl = sum(m.kl_divergence() for m in self.model.modules() if hasattr(m, "kl_divergence"))
        loss += self.kl_weight * kl

        log_test_metrics(self, d_idx, preds, y, mean_sm, loss)
        return {"d_idx": d_idx, "softmax": mean_sm, "label": y, "variance": var_sm}

    def evaluate(self, batch, num_iters: int = 1):

        x, y = batch

        list_out = [self.model(x) for _ in range(num_iters)]

        loss = 0.0
        for out in list_out:
            loss += F.cross_entropy(out, y) / num_iters

        if self.kl_weight != 0.0:
            kl = get_kl_loss(self.model)
            loss = loss + kl / x.size(0)

        out = torch.stack(list_out, dim=0).mean(dim=0)
        out_sm = F.softmax(out, dim=1)

        conf, preds = torch.max(out_sm, dim=1)

        return loss, out_sm, preds, y

    def training_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, out_sm, preds, y = self.evaluate(batch, self.svi_train_samples)
        log_train_metrics(self, d_idx, preds, y, loss)
        return {"d_idx": d_idx, "loss": loss}

    def validation_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, out_sm, preds, y = self.evaluate(batch, self.svi_val_samples)
        log_val_metrics(self, d_idx, preds, y, out_sm, loss)
        return {"d_idx": d_idx, "loss": loss}


@MODEL_REGISTRY
class multiHeadClassifier(EnsembleTileClassifier):

    res_map = {
        18: multiHead_resnet18,
        34: multiHead_resnet34,
        50: multiHead_resnet50,
        101: multiHead_resnet101,
        152: multiHead_resnet152,
    }

    def __init__(
        self,
        ensemble_size: int,
        split_lvl: int,
        size: int,
        num_classes: int,
        loss_function: Union[
            EnsembleLosses.MultiHeadCrossEntropyLoss,
            EnsembleLosses.MulHCELossKernelWeightCos,
        ],
        reduce_fx=None,
        lr: float = 1e-4,
        opti_metric: str = "val_loss",
        patience: int = 3,
        max_num_datasets: int = 2,
        merge_lvl: Optional[int] = None,
    ):

        assert size in multiHeadClassifier.res_map
        net = multiHeadClassifier.res_map[size](ensemble_size, split_lvl, num_classes, reduce_fx, merge_lvl)
        super().__init__(net, loss_function, num_classes, lr=lr, max_num_datasets=max_num_datasets)
