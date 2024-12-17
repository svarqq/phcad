import __init__  # noqa
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import Lambda, Normalize, PILToTensor, Compose, ToTensor
import torch
from torch.utils.data import DataLoader

from phcad.data_handling.spectral_natural_images import SpectralNaturalImages
from phcad.data_handling.utils import get_dataset
from phcad.data_handling.constants import CIFAR10_LABELS, FMNIST_LABELS
from phcad.experiments.spectral_vs_oe import run_spectral_vs_oe


if __name__ == "__main__":
    run_spectral_vs_oe("fmnist", "top")
    run_spectral_vs_oe("cifar10", "airplane")
