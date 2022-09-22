from typing import Optional, Callable, Type, Union, Sequence

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3

from einops import rearrange, reduce, repeat


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Redeclare for import
BasicBlock = BasicBlock
Bottleneck = Bottleneck


class MultiHeadResNet(nn.Module):
    def __init__(
        self,
        num_heads,
        split_lvl,
        block,
        layers,
        num_classes,
        dropout: float = 0.0,
        reduce_fx: Optional[Union[Callable, str]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        merge_lvl: Optional[int] = 1000,
    ):
        super(MultiHeadResNet, self).__init__()

        if merge_lvl is None:
            merge_lvl = 1000

        self.split_lvl = split_lvl
        self.num_heads = num_heads
        self.merge_lvl = merge_lvl

        if merge_lvl is not None:
            assert merge_lvl > split_lvl
            self.latent_space = None

        self.reduce_fx = None
        if reduce_fx is not None:
            if type(reduce_fx) == str:
                if reduce_fx == "mean":
                    self.reduce_fx = lambda x: torch.mean(torch.stack(x, dim=0), dim=0)
                else:
                    raise NotImplementedError()
            else:
                self.reduce_fx = reduce_fx

        shared_part = []
        head_block_lists = [[] for _ in range(num_heads)]
        merge_part = []

        # Implemented split levels
        if split_lvl not in [0, 1, 2, 3, 4, 5]:
            print("Split_lvl {}".format(split_lvl) + " is not implemented.")
            raise NotImplementedError

        if split_lvl == 6 and num_heads != 1:
            raise RuntimeError("Cant set split_lvl to 6 and num_heads greater than 1!")

        self.inplanes = 64
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

        def __make_split_layer(level):
            split_here = True if merge_lvl > level >= split_lvl else False
            # _make_layer alters this value
            inplane_c = self.inplanes
            for head_idx in range(num_heads if split_here else 1):
                self.inplanes = inplane_c

                if level == 0:
                    conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
                    bn1 = norm_layer(self.inplanes)
                    relu1 = nn.ReLU(inplace=True)
                    maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                    layer = [conv1, bn1, relu1, maxpool1]

                elif level == 1:
                    layer = self._make_layer(block, 64, layers[0], stride=1)
                    layer = [layer, self.dropout]
                elif level == 2:
                    layer = self._make_layer(block, 128, layers[1], stride=2)
                    layer = [layer, self.dropout]
                elif level == 3:
                    layer = self._make_layer(block, 256, layers[2], stride=2)
                    layer = [layer, self.dropout]
                elif level == 4:
                    layer = self._make_layer(block, 512, layers[3], stride=2)
                    layer = [layer, self.dropout]

                elif level == 5:
                    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                    # linear_out = nn.Conv2d(
                    #    512 * block.expansion, num_classes, kernel_size=1
                    # )
                    flatten = nn.Flatten()
                    linear_out = nn.Linear(512 * block.expansion, num_classes)

                    layer = [avg_pool, flatten, self.dropout, linear_out]

                if level < split_lvl:
                    shared_part.extend(layer)
                elif merge_lvl > level >= split_lvl:
                    head_block_lists[head_idx].extend(layer)
                else:
                    merge_part.extend(layer)

        __make_split_layer(0)  # Input layer
        __make_split_layer(1)  # Block 1
        __make_split_layer(2)  # Block 2
        __make_split_layer(3)  # Block 3
        __make_split_layer(4)  # Block 4
        __make_split_layer(5)  # Output layer

        # Turn into nn.Sequential
        self.shared_block = nn.Sequential(*shared_part) if len(shared_part) > 0 else None
        self.merge_block = nn.Sequential(*merge_part) if len(merge_part) > 0 else None
        self.headblocks = nn.ModuleList()

        for layers in head_block_lists:

            if len(layers) == 1:
                self.headblocks.append(layers[0])
            elif len(layers) > 1:
                self.headblocks.append(nn.Sequential(*layers))
            else:
                if split_lvl != 5:
                    raise RuntimeError("Encountered empty head")

        if self.shared_block is not None:
            self.shared_block.apply(_weights_init)

        if self.merge_block is not None:
            self.merge_block.apply(_weights_init)

        for sub_mod in self.headblocks:
            sub_mod.apply(_weights_init)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def getActivations(self):
        return self.latent_space

    def getEnsembleParams(self, filter_types: Optional[Sequence] = (torch.nn.Conv2d,)) -> Union[list, list[list]]:

        if filter_types is None:
            params = []

            for member in range(0, self.num_heads):
                tmp = [b.flatten() for b in list(self.headblocks[member].parameters())]
                params.append(torch.cat(tmp))

            return params

        else:
            module_parameters = [[] for _ in range(self.num_heads)]

            for module in zip(*[self.headblocks[i].modules() for i in range(self.num_heads)]):
                if not type(module[0]) in filter_types:
                    continue

                for j in range(self.num_heads):
                    module_parameters[j].append(next(module[j].parameters()))

            return module_parameters

    def forward(self, x):

        outs = []
        out = x

        if self.shared_block is not None:
            out = self.shared_block(out)

        if len(self.headblocks) > 0:
            for i in range(self.num_heads):
                out_c = out.clone()
                out_c = self.headblocks[i](out_c)
                outs.append(out_c)
        else:
            outs.append(out)

        self.latent_space = []

        for i in range(len(outs)):

            if self.merge_block is not None:
                self.latent_space.append(outs[i].clone())
                outs[i] = self.merge_block(outs[i])
            else:
                self.latent_space = None

            # if outs[i].dim() == 4:
            #    outs[i] = rearrange(outs[i], "b c 1 1 -> b c")

        if self.reduce_fx is not None:
            outs = self.reduce_fx(outs)

        return outs


# def resnet18(num_classes, dropout=0.0) -> MultiHeadResNet:
#     return multiHead_resnet18(
#         num_heads=1, split_lvl=0, num_classes=num_classes, reduce_fx=None, merge_lvl=None, dropout=dropout
#     )


# def resnet34(num_classes, dropout=0.0) -> MultiHeadResNet:
#     return multiHead_resnet34(
#         num_heads=1, split_lvl=0, num_classes=num_classes, reduce_fx=None, merge_lvl=None, dropout=dropout
#     )


# def resnet50(num_classes, dropout=0.0) -> MultiHeadResNet:
#     return multiHead_resnet50(
#         num_heads=1, split_lvl=0, num_classes=num_classes, reduce_fx=None, merge_lvl=None, dropout=dropout
#     )


# def resnet101(num_classes, dropout=0.0) -> MultiHeadResNet:
#     return multiHead_resnet101(
#         num_heads=1, split_lvl=0, num_classes=num_classes, reduce_fx=None, merge_lvl=None, dropout=dropout
#     )


# def resnet152(num_classes, dropout=0.0) -> MultiHeadResNet:
#     return multiHead_resnet152(
#         num_heads=1, split_lvl=0, num_classes=num_classes, reduce_fx=None, merge_lvl=None, dropout=dropout
#     )


def multiHead_resnet18(
    num_heads, split_lvl, num_classes, reduce_fx=None, merge_lvl=None, dropout=0.0
) -> MultiHeadResNet:
    return MultiHeadResNet(
        num_heads,
        split_lvl,
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        dropout,
        reduce_fx,
        merge_lvl=merge_lvl,
    )


def multiHead_resnet34(
    num_heads, split_lvl, num_classes, reduce_fx=None, merge_lvl=None, dropout=0.0
) -> MultiHeadResNet:
    return MultiHeadResNet(
        num_heads,
        split_lvl,
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        dropout,
        reduce_fx,
        merge_lvl=merge_lvl,
    )


def multiHead_resnet50(
    num_heads, split_lvl, num_classes, reduce_fx=None, merge_lvl=None, dropout=0.0
) -> MultiHeadResNet:
    return MultiHeadResNet(
        num_heads,
        split_lvl,
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        dropout,
        reduce_fx,
        merge_lvl=merge_lvl,
    )


def multiHead_resnet101(
    num_heads, split_lvl, num_classes, reduce_fx=None, merge_lvl=None, dropout=0.0
) -> MultiHeadResNet:
    return MultiHeadResNet(
        num_heads,
        split_lvl,
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
        dropout,
        reduce_fx,
        merge_lvl=merge_lvl,
    )


def multiHead_resnet152(
    num_heads, split_lvl, num_classes, reduce_fx=None, merge_lvl=None, dropout=0.0
) -> MultiHeadResNet:
    return MultiHeadResNet(
        num_heads,
        split_lvl,
        Bottleneck,
        [3, 8, 36, 3],
        num_classes,
        dropout,
        reduce_fx,
        merge_lvl=merge_lvl,
    )


if __name__ == "__main__":
    net = multiHead_resnet152(3, 2, 2, None, 5)

    module_params = net.getEnsembleParams()
    a = torch.zeros([3, 3, 224, 224])

    out = net(a)
