from collections import OrderedDict

from torch import nn

from phcad.models.utils import construct_decoder


class AEMvTec(nn.Module):
    # The autoencoder used for objects in the mvtec paper
    def __init__(self, **kwargs):
        super(AEMvTec, self).__init__()
        self.phcal = False

        layers = OrderedDict()
        layers["conv1"] = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
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
        decoder = construct_decoder(encoder)
        self.layers = nn.Sequential(
            OrderedDict((("encoder", encoder), ("decoder", decoder)))
        )

    def forward(self, x):
        return self.layers(x)

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
