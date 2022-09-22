import torch
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from typing import *
import torch.nn as nn

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


class SVIResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        inference: str = "ffg",
    ):

        super(SVIResNet, self).__init__()
        self.net = ResNet(block=block, layers=layers, num_classes=num_classes)

        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Flipout",  # Flipout or Reparameterization
            "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }

        dnn_to_bnn(self.net, const_bnn_prior_parameters)

    def forward(self, x):

        out = self.net(x)

        return out


def svi_resnet18(num_classes: int, inference: str = "ffg") -> SVIResNet:
    return SVIResNet(BasicBlock, [2, 2, 2, 2], num_classes, inference)


def svi_resnet34(num_classes: int, inference: str = "ffg") -> SVIResNet:
    return SVIResNet(BasicBlock, [3, 4, 6, 3], num_classes, inference)


def svi_resnet50(num_classes: int, inference: str = "ffg") -> SVIResNet:
    return SVIResNet(Bottleneck, [3, 4, 6, 3], num_classes, inference)


def svi_resnet101(num_classes: int, inference: str = "ffg") -> SVIResNet:
    return SVIResNet(Bottleneck, [3, 4, 23, 3], num_classes, inference)


def svi_resnet152(num_classes: int, inference: str = "ffg") -> SVIResNet:
    return SVIResNet(Bottleneck, [3, 8, 36, 3], num_classes, inference)
