import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


class UniformNoiseImages(Dataset):
    def __init__(self, n_samples, cwh_shape, mean, std):
        self.n_samples = n_samples
        norm = transforms.Normalize(mean, std)
        self.data = norm(
            torch.randint(
                0,
                256,
                [n_samples] + list(cwh_shape),
                dtype=torch.get_default_dtype(),
            )
        )

    def __getitem__(self, idx):
        return self.data[idx], torch.zeros(self.data.shape[-2:])

    def __len__(self):
        return self.n_samples
