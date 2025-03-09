import logging
import json

import torch
import torchvision.transforms.v2.functional as F
import numpy as np

from phcad.test.utils import check_cal_curves

logger = logging.getLogger(__name__)


def calibration_curve(
    inputs_to_indist_pests,
    test_loader,
    n_bins=10,
    localization=False,
    modules=None,
    savepath=None,
    device=None,
):
    # Credit to sklearn: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py
    # and pycalib https://github.com/classifier-calibration/PyCalib/blob/master/pycalib/utils.py
    if savepath and savepath.exists():
        results_generated = check_cal_curves(savepath)
        if results_generated:
            return
    logger.info(f"Getting calibration curve, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"

    for module in modules:
        module.eval()
        module.to(device)
    indist_pests, targets = [], []
    for data, batch_targets in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            batch_scores = inputs_to_indist_pests(data).to("cpu")
            indist_pests.append(batch_scores)
            if localization:
                batch_targets = F.resize(batch_targets, data.shape[-2:])
            targets.append(-batch_targets + 1)
    for module in modules:
        module.to("cpu")

    indist_pests = torch.cat((*indist_pests,)).view(-1).numpy()
    targets = torch.cat((*targets,)).view(-1).numpy()

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_idx = np.digitize(indist_pests, bins) - 1
    bin_pred = np.bincount(bin_idx, weights=indist_pests, minlength=n_bins)
    bin_true = np.bincount(bin_idx, weights=targets, minlength=n_bins)
    bin_total = np.bincount(bin_idx, minlength=n_bins)

    results = {
        "total_bin_counts": bin_total.tolist(),
        "indist_bin_counts": bin_true.tolist(),
        "summed_bin_pests": bin_pred.tolist(),
    }
    with open(savepath, "w") as f:
        f.write(json.dumps(results, indent=2))
