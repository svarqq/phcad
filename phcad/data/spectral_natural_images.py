import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as F


class SpectralNaturalImages(Dataset):
    def __init__(
        self,
        imshape: int,
        localization_targets: bool = False,
        target: int = 1,
        transform=None,
        nsamps: int = 0,
    ):
        if imshape[1] != imshape[2]:
            raise ValueError(f"Bad shape {imshape}, images must be square")
        if target != 0 and target != 1:
            raise ValueError(f"Target must be 0 or 1, currently target={target}")
        self.imshape = imshape
        self.nsamps = nsamps
        self.localization_targets = localization_targets
        self.target = torch.tensor(target, dtype=torch.get_default_dtype())
        self.transform = transform
        if nsamps:
            self.generate_static_data()

    def __len__(self):
        return self.nsamps if self.nsamps else 2 << 16

    def __getitem__(self, idx):
        if self.nsamps:
            im = self.ims[idx]
        else:
            im = generate_natural_image_from_spectrum(self.imshape[-1], self.imshape[0])

        if self.localization_targets:
            if self.target == 1:
                target = torch.ones(self.imshape[-2:], dtype=torch.uint8)
            elif self.target == 0:
                target = torch.zeros(self.imshape[-2:], dtype=torch.uint8)
        else:
            target = self.target

        if self.transform:
            im = self.transform(im)
        return im, target

    def switch_target_type(self):
        self.localization_targets = not self.localization_targets

    def generate_static_data(self):
        self.ims = [
            generate_natural_image_from_spectrum(self.imshape[-1], self.imshape[0])
            for _ in range(self.nsamps)
        ]


def generate_natural_image_from_spectrum(
    imlen: int = 224, channels=3, alpha_low=0.5, alpha_high=3.5
):
    if channels != 3 and channels != 1:
        raise ValueError(
            f"Number of channels must be 3 (RGB) or 1 (GS), got argument channels={channels}"
        )

    alpha_center = alpha_low + torch.rand(1) * (alpha_high - alpha_low)
    offset = torch.randn(1) * (alpha_high - alpha_low) / 60
    alpha_x, alpha_y = alpha_center + offset, alpha_center - offset
    fx, fy = torch.meshgrid(*(torch.arange(imlen),) * 2, indexing="xy")
    fx, fy = map(lambda f: (f - imlen / 2) / imlen, [fx, fy])
    f_magnitude = 1 / torch.fft.fftshift(
        1e-16 + (torch.abs(fx) ** alpha_x + torch.abs(fy) ** alpha_y)
    )
    f_magnitude[0, 0] = 0

    im = torch.rand((channels, imlen, imlen)) * 255
    for c in range(channels):
        uniform_noise = im[c]
        random_phase = torch.angle(torch.fft.fft2(uniform_noise - uniform_noise.mean()))

        ch_spectrum = f_magnitude * torch.exp(1j * random_phase)
        ch_pixels = torch.real(torch.fft.ifft2(ch_spectrum))
        ch_pixels = (
            ch_pixels - ch_pixels.mean() / ch_pixels.std()
        ) * uniform_noise.std() + uniform_noise.mean()
        im[c] = ch_pixels

    im = ((im - im.min()) / (im.max() - im.min()) * 255).to(torch.uint8)
    return F.to_pil_image(im)
