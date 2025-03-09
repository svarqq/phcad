import __init__  # noqa
import logging
import argparse

from phcad.data.constants import DS_TO_LABELS_MAP
from phcad.train.losses import SEG_LOSS_MAP
from phcad.experiments.localization_calibration import get_seg_cal_curves

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name")
parser.add_argument("loss_name")
parser.add_argument("label_idx")
parser.add_argument("--spectral-train", action="store_true")
parser.add_argument("--spectral-cal", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset_name not in DS_TO_LABELS_MAP.keys():
        raise ValueError(f"dataset_name must be one of {list(DS_TO_LABELS_MAP.keys())}")
    dname = args.dataset_name
    if args.loss_name not in SEG_LOSS_MAP.keys():
        raise ValueError(f"loss_name must be one of {list(SEG_LOSS_MAP.keys())}")
    max_idx = len(DS_TO_LABELS_MAP[dname]) - 1
    lidx = int(args.label_idx)
    if lidx < 0 or lidx > max_idx:
        raise ValueError(f"For {dname}, label_idx must be between 0 and {max_idx}")

    label = DS_TO_LABELS_MAP[dname][lidx]
    xargs = {
        "dataset_name": dname,
        "label": label,
        "loss_name": args.loss_name,
        "spectral_oe_train": args.spectral_train,
        "spectral_oe_cal": args.spectral_cal,
    }
    logging.info(f"Getting seg calibration curve for label {label} of {dname}")
    get_seg_cal_curves(**xargs)
