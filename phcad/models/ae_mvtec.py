from collections import OrderedDict

import torch
from torch import nn

from phcad.models.utils import construct_decoder


class AEMvTec(nn.Module):
    # The autoencoder used for objects in the mvtec paper
    def __init__(self, unet=False, **kwargs):
        super(AEMvTec, self).__init__()
        self.phcal = False
        self.unet = unet
        self.encoder_skip_layernums = [0, 3, 5, 6, 7, 8]

        if unet:
            # U-Net style with BCE
            nch = 3
        else:
            # Vanilla autoencoder with SSIM
            nch = 1

        layers = OrderedDict()
        layers["conv1"] = nn.Sequential(
            nn.Conv2d(nch, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
        layers["conv2"] = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        layers["conv3"] = nn.Sequential(
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        layers["conv4"] = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        layers["conv5"] = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        layers["conv6"] = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        layers["conv7"] = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        layers["conv8"] = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        layers["conv9"] = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        layers["conv10"] = nn.Sequential(
            nn.Conv2d(32, 100, 8, stride=1),
        )
        encoder = nn.Sequential(layers)
        decoder = construct_decoder(encoder, unet=unet)
        self.layers = nn.Sequential(
            OrderedDict((("encoder", encoder), ("decoder", decoder)))
        )

    def forward(self, x):
        if not self.unet:
            return self.layers(x)
        else:
            encoder_outputs = [0] * len(self.layers.encoder)
            for layernum, layer in enumerate(self.layers.encoder):
                x = layer(x)
                if layernum in self.encoder_skip_layernums:
                    encoder_outputs[layernum] = x
            for layernum, layer in enumerate(self.layers.decoder):
                corresponding_encoder_layernum = len(self.layers.decoder) - layernum - 1
                add_encoder_outputs = (
                    corresponding_encoder_layernum in self.encoder_skip_layernums
                )
                if add_encoder_outputs:
                    x = layer(x + encoder_outputs[corresponding_encoder_layernum])
                else:
                    x = layer(x)
            return x.squeeze(-3)

    def prepare_calibration_network(self):
        if self.phcal:
            raise Exception("Network already set up for post-hoc calibration")

        calibration_head = OrderedDict()
        calibration_head["pre-fc"] = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(100), nn.LeakyReLU()
        )
        calibration_head["fc1"] = nn.Sequential(nn.Linear(100, 1), nn.Flatten(0, 1))
        layers = nn.Sequential(
            OrderedDict(
                (
                    ("encoder", self.layers.encoder),
                    ("calibration_head", nn.Sequential(calibration_head)),
                )
            )
        )
        layers.encoder.requires_grad_(False)
        self.layers = layers
        self.phcal = True
