from collections import OrderedDict
import tomllib
import torch

from phcad.models.layers import PerPixelPlattCal
from phcad.constants import ARCHDIR


class AEMvTec(torch.nn.Module):
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
            conv = torch.nn.Conv2d(**conv_layer["args"])
            enc_unit.update({"conv": conv})
            if not last_layer:
                enc_unit.update(
                    {"bn": torch.nn.BatchNorm2d(conv_layer["args"]["out_channels"])}
                )
                enc_unit.update({"activ": torch.nn.LeakyReLU(0.2)})
            encoder.append(torch.nn.Sequential(enc_unit))

            # Mirror encoder to construct decoder
            dec_unit = OrderedDict()
            tconv_layer = conv_layer.copy()
            tconv_layer["args"]["in_channels"], tconv_layer["args"]["out_channels"] = (
                tconv_layer["args"]["out_channels"],
                tconv_layer["args"]["in_channels"],
            )
            tconv = torch.nn.ConvTranspose2d(**tconv_layer["args"])
            dec_unit.update({"tconv": tconv})
            if not first_layer:
                dec_unit.update(
                    {"bn": torch.nn.BatchNorm2d(tconv_layer["args"]["out_channels"])}
                )
                dec_unit.update({"activ": torch.nn.LeakyReLU(0.2)})
            decoder = [torch.nn.Sequential(dec_unit)] + decoder
        self.encoder = torch.nn.Sequential(*encoder)
        self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):
        recon = self.decoder(self.encoder(x))
        try:
            if not self.cal:
                return recon
            else:
                recon_across_channels = recon.mean(1)
                return self.cal(recon_across_channels)
        except AttributeError:
            return recon

    def setup_cal(self, wh_shape, head_layers_to_reset=0):
        try:
            if self.cal:
                raise RuntimeError("Model already setup for calibration")
        except AttributeError:
            pass
        self.requires_grad_(False)

        head_layers = []
        for i in range(1, head_layers_to_reset + 1):
            layer = list(self.decoder.children())[-i]
            layer.requires_grad_(True)

            def reset(child):
                try:
                    child.reset_parameters()
                except AttributeError:
                    pass

            layer.apply(reset)
            head_layers.append(layer)

        self.cal = PerPixelPlattCal(wh_shape)
        head_layers.append(self.cal)
        return head_layers


if __name__ == "__main__":
    ae = AEMvTec(gray=False)
    print(ae)
