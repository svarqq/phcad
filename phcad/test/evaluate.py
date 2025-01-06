import logging
import json

import torch
from torchvision.transforms.v2 import Normalize
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_thresholding(
    inputs_to_anomaly_score,
    test_loader,
    modules=None,
    savepath=None,
    device=None,
):
    if savepath and savepath.exists():
        logger.info(
            f"Results already saved to {savepath}, delete if needed to run again"
        )
        return
    logger.info(f"Getting results, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"
    for module in modules:
        module.eval()
        module.to(device)
    anomaly_scores, targets = [], []
    for data, batch_targets in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            batch_scores = inputs_to_anomaly_score(data).to("cpu")
            anomaly_scores.append(batch_scores)
            targets.append(batch_targets)
    for module in modules:
        module.to("cpu")

    anomaly_scores = torch.cat((*anomaly_scores,)).numpy()
    targets = torch.cat((*targets,)).numpy()

    auroc = roc_auc_score(targets, anomaly_scores)
    roc = roc_curve(targets, anomaly_scores, drop_intermediate=False)[:2]
    ap = average_precision_score(targets, anomaly_scores)
    prc = precision_recall_curve(targets, anomaly_scores, drop_intermediate=False)[:2]

    logger.info(f"AUROC: {auroc}")
    logger.info(f"AP: {ap}")

    results = {
        "auroc": auroc,
        "ap": ap,
        "roc": {"x": list(roc[0]), "y": list(roc[1])},
        "prc": {"x": list(prc[0]), "y": list(prc[1])},
    }
    with open(savepath, "w") as f:
        f.write(json.dumps(results, indent=2))
    return


def evaluate_thresholding_perturbation(
    inputs_to_anomaly_score,
    inputs_to_loss,
    norm_std,
    test_loader,
    modules=None,
    savepath=None,
    device=None,
    eps=0.0014,
):
    if savepath and savepath.exists():
        logger.info(
            f"Results already saved to {savepath}, delete if needed to run again"
        )
        return
    logger.info(f"Getting results with perturbation, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"

    norm = Normalize([0.0] * len(norm_std), norm_std)
    for module in modules:
        module.eval()
        module.to(device)
        module.requires_grad_(True)
    anomaly_scores, targets = [], []
    for data, batch_targets in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            with torch.enable_grad():
                data.requires_grad = True
                batch_targets = batch_targets.to(device)
                loss = inputs_to_loss(data, batch_targets)
                loss.backward()
            signs = torch.ge(data.grad, 0) * 2 - 1
            perturbations = norm(signs * eps)

            perturbed_data = data - perturbations
            batch_scores = inputs_to_anomaly_score(perturbed_data)
            anomaly_scores.append(batch_scores.to("cpu").detach())
            targets.append(batch_targets.to("cpu").detach())
    for module in modules:
        module.to("cpu")

    anomaly_scores = torch.cat((*anomaly_scores,)).numpy()
    targets = torch.cat((*targets,)).numpy()

    auroc = roc_auc_score(targets, anomaly_scores)
    roc = roc_curve(targets, anomaly_scores, drop_intermediate=False)[:2]
    ap = average_precision_score(targets, anomaly_scores)
    prc = precision_recall_curve(targets, anomaly_scores, drop_intermediate=False)[:2]

    logger.info(f"AUROC: {auroc}")
    logger.info(f"AP: {ap}")

    results = {
        "auroc": auroc,
        "ap": ap,
        "roc": {"x": list(roc[0]), "y": list(roc[1])},
        "prc": {"x": list(prc[0]), "y": list(prc[1])},
    }
    with open(savepath, "w") as f:
        f.write(json.dumps(results, indent=2))
    return

    pass
