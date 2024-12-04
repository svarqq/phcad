from collections import OrderedDict
from math import prod

import torch
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt


class FCDDMvTec(torch.nn.Module):
    def __init__(self, input_2dshape):
        super(FCDDMvTec, self).__init__()
        vgg11_base = torchvision.models.vgg11_bn().features[:-8]
        vgg11_base[:15].requires_grad_(False)

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

        self.model = OrderedDict(
            (
                ("vgg11_bn_base", vgg11_base),
                ("final_conv"),
                Conv2D(512, 1, 1)(
                    "receptive_upsample",
                    ReceptiveUpsample(ksizes, strides, paddings),
                ),
            )
        )


class ReceptiveUpsample(torch.nn.Module):
    def __init__(self, kernel_sizes, strides, paddings, std=10):
        # See https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-size for computing effective receptive field values
        super(ReceptiveUpsample, self).__init__()
        stride_products = [1] + [prod(strides[:i]) for i in range(1, len(strides))]
        print(strides, stride_products)
        self.effective_padding = sum(
            [pad * sp for pad, sp in zip(paddings, stride_products)]
        )
        print(paddings)
        self.effective_stride = prod(strides)
        self.effective_size = (
            sum([(k_sz - 1) * sp for k_sz, sp in zip(kernel_sizes, stride_products)])
            + 1
        )
        print(self.effective_padding, self.effective_size, self.effective_stride)

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
        # plt.imshow(gauss_weights)
        # plt.show()
        im = torch.unsqueeze(
            torch.unsqueeze(
                torch.tensor([[x + y for x in range(28)] for y in range(28)]), 0
            ),
            0,
        ).to(torch.float)
        print(im.shape)
        im2 = F.conv_transpose2d(
            im,
            weight=gauss_weight,
            stride=self.effective_stride,
            padding=self.effective_padding,
        )
        print(im2.shape)
        plt.imshow(im2[0][0])
        plt.show()


if __name__ == "__main__":
    model = FCDDMvTec((224, 224))
