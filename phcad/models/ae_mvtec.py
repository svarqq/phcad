from collections import OrderedDict
import tomllib
from torch import nn

from phcad.constants import ARCHDIR


class AEMvTec(nn.Module):
    def __init__(self):
        super(AEMvTec, self).__init__()

        input_layer_channels = 3  # rgb
        with open(ARCHDIR / "ae-mvtec.toml", "rb") as f:
            enc_conv_layers = tomllib.load(f)["network"]["encoder"]
        encoder, decoder = [], []
        for lnum, conv_layer in enumerate(enc_conv_layers):
            first_layer = lnum == 0
            last_layer = lnum == len(enc_conv_layers) - 1

            if first_layer:
                conv_layer["args"]["in_channels"] = input_layer_channels

            enc_unit = OrderedDict()
            conv = nn.Conv2d(**conv_layer["args"])
            enc_unit.update({"conv": conv})
            if not last_layer:
                enc_unit.update(
                    {"bn": nn.BatchNorm2d(conv_layer["args"]["out_channels"])}
                )
                enc_unit.update({"activ": nn.LeakyReLU(0.2)})
            encoder.append(nn.Sequential(enc_unit))

            # Mirror encoder to construct decoder
            dec_unit = OrderedDict()
            tconv_layer = conv_layer.copy()
            tconv_layer["args"]["in_channels"], tconv_layer["args"]["out_channels"] = (
                tconv_layer["args"]["out_channels"],
                tconv_layer["args"]["in_channels"],
            )
            tconv = nn.ConvTranspose2d(**tconv_layer["args"])
            dec_unit.update({"tconv": tconv})
            if not first_layer:
                dec_unit.update(
                    {"bn": nn.BatchNorm2d(tconv_layer["args"]["out_channels"])}
                )
                dec_unit.update({"activ": nn.LeakyReLU(0.2)})
            decoder = [nn.Sequential(dec_unit)] + decoder
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    ae = AEMvTec(gray=False)
    print(ae)
