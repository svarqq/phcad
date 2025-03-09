import __init__  # noqa

from phcad.experiments.results import parse_results, parse_cal_curves
from phcad.data.constants import DS_TO_LABELS_MAP
from phcad.train.losses import LOSS_MAP, SEG_LOSS_MAP

if __name__ == "__main__":
    for ds in DS_TO_LABELS_MAP.keys():
        if ds == "imagenet30":  # or ds == "cifar10" or ds == "fmnist":
            continue
        for loss in LOSS_MAP.keys():
            parse_cal_curves(ds, loss, "detection")
