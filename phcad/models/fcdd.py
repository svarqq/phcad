from collections import OrderedDict
from math import prod

import torch
import torch.nn.functional as F
from torch.nn.parameter import Buffer
from torchvision import models


class FCDD(torch.nn.Module):
    def __init__(self):
        super(FCDD, self).__init__()
        vgg11_base = models.vgg11_bn(weights="DEFAULT").features[:-8]
        vgg11_base[:15].requires_grad_(False)

        self.layers = torch.nn.Sequential(
            OrderedDict(
                (
                    ("vgg11_bn_base", vgg11_base),
                    ("final_conv", torch.nn.Conv2d(512, 1, 1)),
                )
            )
        )

    def forward(self, x):
        return self.layers(x)


class ReceptiveUpsample(torch.nn.Module):
    def __init__(self, std=14):
        super(ReceptiveUpsample, self).__init__()

        # See https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-size
        # for computing effective receptive field values
        vgg11_base = models.vgg11_bn(weights="DEFAULT").features[:-8]
        ksizes, strides, paddings = [], [], []
        for module in vgg11_base:
            try:
                ksizes.append(
                    module.kernel_size
                    if isinstance(module.kernel_size, int)
                    else module.kernel_size[0]
                )
                strides.append(
                    module.stride
                    if isinstance(module.stride, int)
                    else module.stride[0]
                )
                paddings.append(
                    module.padding
                    if isinstance(module.padding, int)
                    else module.padding[0]
                )
            except AttributeError:
                pass

        stride_products = [1] + [prod(strides[:i]) for i in range(1, len(strides))]
        self.effective_padding = sum(
            [pad * sp for pad, sp in zip(paddings, stride_products)]
        )
        self.effective_stride = prod(strides)
        self.effective_size = (
            sum([(k_sz - 1) * sp for k_sz, sp in zip(ksizes, stride_products)]) + 1
        )

        coordinates = torch.stack(
            torch.meshgrid(*(torch.arange(self.effective_size),) * 2, indexing="xy")
        )
        gauss_weight = torch.unsqueeze(
            torch.unsqueeze(
                (
                    1
                    / (2 * torch.pi * std**2)
                    * torch.exp(
                        -((coordinates - self.effective_size / 2) ** 2).sum(0)
                        / (2 * std**2)
                    )
                ),
                0,
            ),
            0,
        )
        self.weight = Buffer(gauss_weight)

    def forward(self, x):
        return F.conv_transpose2d(
            x,
            weight=self.weight,
            stride=self.effective_stride,
            padding=self.effective_padding,
        )
