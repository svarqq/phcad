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
from phcad.data.spectral_natural_images import SpectralNaturalImages
from phcad.data.mvtec_mpdd import MVTecMPDD
from phcad.data.utils import get_dataset, get_train_cal_splits
from phcad.data.constants import CIFAR10_LABELS, FMNIST_LABELS
from phcad.train.losses import LOSS_MAP
from phcad.experiments.constants import EXPROOT
from phcad.experiments.detection import run_detection_experiment
from phcad.models.layers import PlattCal


if __name__ == "__main__":
    spec_oe_train = False
    spec_oe_cal = False
    for label in FMNIST_LABELS:
        for loss in LOSS_MAP:
            run_detection_experiment("fmnist", label, loss, spec_oe_train, False)
            run_detection_experiment("fmnist", label, loss, spec_oe_train, True)
    for label in CIFAR10_LABELS:
        for loss in LOSS_MAP:
            run_detection_experiment("cifar10", label, loss, spec_oe_train, False)
            run_detection_experiment("cifar10", label, loss, spec_oe_train, True)

    # wrn_open = wrn18.WideResNet18()
    # wrn_clf = wrn18.WideResNet18(clf=True)
    # ae = ae_mvtec.AEMvTec()

    # breakpoint()
