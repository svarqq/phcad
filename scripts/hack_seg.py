import __init__  # noqa
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import Lambda, Normalize, PILToTensor, Compose, ToTensor
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from phcad.models import wrn18, ae_mvtec
from phcad.data_handling.constants import DATADIR
from phcad.data_handling.spectral_natural_images import SpectralNaturalImages
from phcad.data_handling.mvtec_mpdd import MVTecMPDD
from phcad.data_handling.utils import get_dataset, get_train_cal_splits
from phcad.data_handling.constants import CIFAR10_LABELS, FMNIST_LABELS
from phcad.data_handling.transforms import (
    get_default_test_transform,
    get_default_train_transform,
)
from phcad.experiments.constants import EXPDIR
from phcad.experiments.segmentation import run_segmentation_experiment
from phcad.models.layers import PlattCal


if __name__ == "__main__":
    spec_oe_train = False
    spec_oe_cal = False

    # run_segmentation_experiment("mvtec", "bottle", "fcdd", spec_oe_train, spec_oe_cal)

    # TODO: BUG PRESENT
    run_segmentation_experiment(
        "mpdd", "bracket_brown", "bce", spec_oe_train, spec_oe_cal
    )

    # td = MVTecMPDD("mpdd", DATADIR / "mpdd", "bracket_brown", train=False)
    # td.transform = get_default_train_transform([0, 0, 0], [1, 1, 1])
    # tmp = DataLoader(td, batch_size=128)
    # for d in tmp:
    #    d[0][1]

    # wrn_open = wrn18.WideResNet18()
    # wrn_clf = wrn18.WideResNet18(clf=True)
    # ae = ae_mvtec.AEMvTec()

    # breakpoint()
