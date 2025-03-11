from collections import OrderedDict

from torch import nn

from phcad.models.layers import LinearActivation


def construct_decoder(encoder, unet=False, bias=True, eps=1e-05):
    decoder = OrderedDict()
    encoder_children = list(encoder.named_children())

    tconv_num, upsamp_num = 1, 1
    for i, unit in enumerate(reversed(encoder_children)):
        name, layer = unit
        if "conv" in name:
            in_channels, out_channels, kernel_size, stride, padding = [None] * 5
            for child in layer.children():
                if child.__class__ == nn.Conv2d:
                    in_channels, out_channels = child.out_channels, child.in_channels
                    kernel_size, stride = child.kernel_size, child.stride
                    try:
                        padding = child.padding
                    except AttributeError:
                        pass

            if i == len(encoder_children) - 1 and unet:
                out_channels = 1
            tconv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding if padding else 0,
                bias=bias,
            )

            if i != len(encoder_children) - 1:
                decoder[f"tconv{tconv_num}"] = nn.Sequential(
                    tconv,
                    nn.BatchNorm2d(out_channels, eps=eps, affine=bias),
                    nn.LeakyReLU(),
                )
            else:
                decoder[f"tconv{tconv_num}"] = nn.Sequential(tconv, LinearActivation())
            tconv_num += 1

        elif "pool" in name:
            decoder[f"upsamp{upsamp_num}"] = nn.Upsample(scale_factor=2)
            upsamp_num += 1

    decoder = nn.Sequential(decoder)
    return decoder
