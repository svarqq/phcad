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

from phcad.models.cnn_cifar10 import CNN_CIFAR10
from phcad.data_handling.spectral_natural_images import SpectralNaturalImages
from phcad.data_handling.mvtec_mpdd import MVTecMPDD
from phcad.data_handling.utils import get_dataset, get_train_cal_splits
from phcad.data_handling.constants import CIFAR10_LABELS, FMNIST_LABELS
from phcad.experiments.constants import EXPDIR
from phcad.experiments.onevall import run_onevall
from phcad.models.layers import PlattCal


if __name__ == "__main__":
    spec_oe_train = False
    spec_oe_cal = False
    run_onevall("fmnist", "ankle-boot", "ssim", spec_oe_train, spec_oe_cal)

    # data = get_dataset("mvtec", "test", "bottle")
    # for im, mask in data:
    #    if np.all(mask.numpy() == 0):
    #        print("Found indist")
    #    elif not np.all(mask.numpy() != 0) and not np.all(mask.numpy() != 1):
    #        print("Found anom")
    #    else:
    #        print(np.unique(mask.numpy()))
    #        print(np.all(mask.numpy() != 0), np.all(mask.numpy() != 1))
    #        print("ERRORRERROR")
    #        break

    # model = CNN_CIFAR10(clf=True)
    # print(model)
    # model.prepare_calibration_network()
    # print(model)
