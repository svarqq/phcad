import __init__  # noqa
import logging
import sys

from phcad.experiments.onevall_classification import run_onevall_exp_mvtec_ae


if __name__ == "__main__":
    labels = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    label = labels[int(sys.argv[1])]
    logging.info(f"Training on label {label}")
    run_onevall_exp_mvtec_ae(label)
