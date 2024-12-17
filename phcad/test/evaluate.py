import logging
import json

import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_thresholding(
    model, test_loader, anomaly_score_function, savepath, device=None
):
    if savepath.exists():
        logger.info(
            f"Results already saved to {savepath}, delete if needed to run again"
        )
        return
    logger.info(f"Getting results, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"
    model.eval()
    model.to(device)
    anomaly_scores, targets = [], []
    for data, batch_targets in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            batch_scores = anomaly_score_function(model(data)).to("cpu")
            anomaly_scores.append(batch_scores)
            targets.append(batch_targets)
    model.to("cpu")

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
