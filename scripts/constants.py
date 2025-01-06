import __init__  # noqa
from phcad.constants import SAVEROOT
from phcad.data_handling.constants import (
    CIFAR10_LABELS,
    FMNIST_LABELS,
    IMAGENET30_LABELS,
    MPDD_LABELS,
    MVTEC_LABELS,
)

DS_TO_LABELS_MAP = {
    "cifar10": CIFAR10_LABELS,
    "fmnist": FMNIST_LABELS,
    "imagenet30": IMAGENET30_LABELS,
    "mpdd": MPDD_LABELS,
    "mvtec": MVTEC_LABELS,
}

SLURMDIR = SAVEROOT / "slurm"
