import __init__  # noqa
import sys
import logging
from constants import DS_TO_LABELS_MAP

from phcad.experiments.spectral_vs_oe import run_spectral_vs_oe


if __name__ == "__main__":
    dataset, label_idx = sys.argv[1], int(sys.argv[2])
    label = DS_TO_LABELS_MAP[dataset][label_idx]
    logging.info(f"Started spectral-vs-oe experiment for label {label} of {dataset}")
    run_spectral_vs_oe(dataset, label)
