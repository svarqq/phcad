from collections import OrderedDict

from torch import nn

from phcad.models.layers import LinearActivation
from phcad.models.utils import construct_decoder


class CNN_FMNIST(nn.Module):
    # for 28 x 28 inputs
    def __init__(self, bias=True, clf=False, ae=False):
        if clf and ae:
            raise ValueError(
                "clf and ae can't both be true -- i.e. network can't output one value for classification and still be an autoencoder"
            )

        super(CNN_FMNIST, self).__init__()
        self.clf = clf
        self.ae = ae
        self.phcal = False

        self.eps = 1e-04
        self.bias = bias

        layers = OrderedDict()
        layers["conv1"] = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2, bias=self.bias),
            nn.BatchNorm2d(16, eps=self.eps, affine=self.bias),
            nn.LeakyReLU(),
        )
        layers["pool1"] = nn.MaxPool2d(2, 2)
        layers["conv2"] = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2, bias=self.bias),
            nn.BatchNorm2d(32, eps=self.eps, affine=self.bias),
            nn.LeakyReLU(),
        )
        layers["pool2"] = nn.MaxPool2d(2, 2)

        if not ae:
            layers["fc-flatten"] = nn.Flatten()
            layers["fc1"] = nn.Sequential(
                nn.Linear(32 * 7 * 7, 64, bias=self.bias),
                nn.BatchNorm1d(64, eps=self.eps, affine=self.bias),
                nn.LeakyReLU(),
            )
            if not clf:
                layers["fc2"] = nn.Sequential(
                    nn.Linear(64, 32, bias=self.bias), LinearActivation()
                )
            else:
                self.layers = nn.Sequential(layers)
                layers["fc2"] = nn.Sequential(
                    nn.Linear(64, 32, bias=self.bias),
                    nn.BatchNorm1d(32, eps=self.eps, affine=self.bias),
                    nn.LeakyReLU(),
                )
                layers["fc3"] = nn.Sequential(
                    nn.Linear(32, 1, bias=self.bias),
                    nn.Flatten(0, 1),
                    LinearActivation(),
                )
            self.layers = nn.Sequential(layers)

        else:
            layers["conv3"] = nn.Sequential(
                nn.Conv2d(32, 100, 7, bias=self.bias), LinearActivation()
            )
            encoder = nn.Sequential(layers)
            decoder = construct_decoder(encoder, self.bias, self.eps)
            self.layers = nn.Sequential(
                OrderedDict((("encoder", encoder), ("decoder", decoder)))
            )

    def forward(self, x):
        return self.layers(x)

    def prepare_calibration_network(self):
        # Bias can be set to false to accommodate DSVDD, but in post-hoc calibration
        # it can be turned back on
        if self.phcal:
            raise Exception("Network already set up for post-hoc calibration")

        calibration_head = OrderedDict()
        if not self.clf and not self.ae:
            # Remove linear activation
            base_layers = OrderedDict(self.layers.named_children())
            base_layers["fc2"] = nn.Sequential(
                *(list(base_layers["fc2"].children())[:-1])
            )
            base_layers = nn.Sequential(OrderedDict(base_layers))

            calibration_head["fc2-post"] = nn.Sequential(
                nn.BatchNorm1d(32, eps=self.eps), nn.LeakyReLU()
            )
            calibration_head["fc3"] = nn.Sequential(
                nn.Linear(32, 1), nn.Flatten(0, 1), LinearActivation()
            )
            layers = nn.Sequential(base_layers, nn.Sequential(calibration_head))

        elif self.ae:
            # Remove linear activation
            base_layers = OrderedDict(self.layers.encoder.named_children())
            base_layers["conv3"] = nn.Sequential(
                *(list(base_layers["conv3"].children())[:-1])
            )
            base_layers = nn.Sequential(OrderedDict(base_layers))

            calibration_head["pre-fc"] = nn.Sequential(
                nn.Flatten(), nn.BatchNorm1d(100, eps=self.eps), nn.LeakyReLU()
            )
            calibration_head["fc1"] = nn.Sequential(
                nn.Linear(100, 1), nn.Flatten(0, 1), LinearActivation()
            )
            self.ae = False

        elif self.clf:
            base_layers = OrderedDict(self.layers.named_children())
            calibration_head.update((base_layers.popitem(),))
            base_layers = nn.Sequential(OrderedDict(base_layers))

        layers = nn.Sequential(
            OrderedDict(
                (
                    ("base_layers", base_layers),
                    ("calibration_head", nn.Sequential(calibration_head)),
                )
            )
        )
        layers.base_layers.requires_grad_(False)
        self.layers = layers
        self.clf = True
        self.phcal = True
