from phcad.data_handling.constants import FMNIST_LABELS
from phcad.data_handling.utils import get_dataset, BalancedLoader
from phcad.data_handling.spectral_natural_images import SpectralNaturalImages


def run_fmnist_experiments():
    for label in FMNIST_LABELS:
        fmnist_train = get_dataset("fmnist", "train", label)
        spectral_anom = SpectralNaturalImages(fmnist_train[0][0].size[0])
