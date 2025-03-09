import logging
import json

import torch
from torchvision.transforms.v2 import Normalize
import torchvision.transforms.v2.functional as F
from torchmetrics.utilities.compute import auc
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from anomalib.metrics import AUROC, AUPRO

from phcad.data.transforms import mask_to_class
from phcad.test.utils import check_results

logger = logging.getLogger(__name__)


def evaluate_thresholding(
    inputs_to_anomaly_score,
    test_loader,
    modules=None,
    savepath=None,
    device=None,
):
    if savepath and savepath.exists():
        results_generated = check_results(savepath)
        if results_generated:
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
        results_generated = check_results(savepath)
        if results_generated:
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


def evaluate_thresholding_localization(
    inputs_to_anomaly_map,
    test_loader,
    modules=None,
    savepath=None,
    device=None,
    gen_aupro=True,
):
    if savepath and savepath.exists():
        results_generated = check_results(savepath, check_aupro=gen_aupro)
        if results_generated:
            return
    logger.info(f"Getting results, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"

    for module in modules:
        module.eval()
        module.to(device)
    aupro_obj, auroc_obj = AUPRO(fpr_limit=0.3), AUROC()
    for data, batch_target_masks in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            batch_maps = inputs_to_anomaly_map(data)
            rsz_masks = F.resize(batch_target_masks, data.shape[-2:])
            rsz_masks[rsz_masks >= 0.5] = 1
            rsz_masks[rsz_masks < 0.5] = 0
        auroc_obj.update(
            batch_maps.to("cpu").view(-1), rsz_masks.to(torch.long).to("cpu").view(-1)
        )
        aupro_obj.update(batch_maps.to("cpu"), rsz_masks.to(torch.long).to("cpu"))
    for module in modules:
        module.to("cpu")

    roc = auroc_obj._compute()
    auroc = auc(roc[0], roc[1], reorder=True)
    if gen_aupro:
        pro = aupro_obj._compute()
        aupro = auc(pro[0], pro[1], reorder=True)
        aupro /= pro[0][-1]

    logger.info(f"AUROC: {auroc}")
    if gen_aupro:
        logger.info(f"AUPRO: {aupro}")

    if gen_aupro:
        results = {
            "auroc": float(auroc),
            "aupro": float(aupro),
            "roc": {"x": roc[0].tolist(), "y": roc[1].tolist()},
            "pro": {"x": pro[0].tolist(), "y": pro[1].tolist()},
        }
    else:
        results = {
            "auroc": float(auroc),
            "roc": {"x": roc[0].tolist(), "y": roc[1].tolist()},
        }
    with open(savepath, "w") as f:
        f.write(json.dumps(results, indent=2))
    return


def evaluate_thresholding_localization_perturbation(
    inputs_to_anomaly_map,
    inputs_to_loss,
    norm_std,
    test_loader,
    detection_targets_for_loss=False,
    modules=None,
    savepath=None,
    device=None,
    eps=0.0014,
    gen_aupro=True,
):
    if savepath and savepath.exists():
        results_generated = check_results(savepath, check_aupro=gen_aupro)
        if results_generated:
            return
    logger.info(f"Getting results, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"

    norm = Normalize([0.0] * len(norm_std), norm_std)
    for module in modules:
        module.eval()
        module.to(device)
        module.requires_grad_(True)
    aupro_obj, auroc_obj = AUPRO(fpr_limit=0.3), AUROC()
    for data, batch_target_masks in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)

            rsz_masks = F.resize(batch_target_masks, data.shape[-2:])
            rsz_masks[rsz_masks >= 0.5] = 1
            rsz_masks[rsz_masks < 0.5] = 0
            if detection_targets_for_loss:
                targets = torch.stack(list(map(mask_to_class, batch_target_masks)))
            else:
                targets = rsz_masks
            targets = targets.to(device)

            with torch.enable_grad():
                data.requires_grad = True
                loss = inputs_to_loss(data, targets)
                loss.backward()
            signs = torch.ge(data.grad, 0) * 2 - 1
            perturbations = norm(signs * eps)

            perturbed_data = data - perturbations
            batch_maps = inputs_to_anomaly_map(perturbed_data)
        auroc_obj.update(
            batch_maps.to("cpu").view(-1), rsz_masks.to(torch.long).to("cpu").view(-1)
        )
        aupro_obj.update(batch_maps.to("cpu"), rsz_masks.to(torch.long).to("cpu"))
    for module in modules:
        module.to("cpu")

    roc = auroc_obj._compute()
    auroc = auc(roc[0], roc[1], reorder=True)
    if gen_aupro:
        pro = aupro_obj._compute()
        aupro = auc(pro[0], pro[1], reorder=True)
        aupro /= pro[0][-1]

    logger.info(f"AUROC: {auroc}")
    if gen_aupro:
        logger.info(f"AUPRO: {aupro}")

    if gen_aupro:
        results = {
            "auroc": float(auroc),
            "aupro": float(aupro),
            "roc": {"x": roc[0].tolist(), "y": roc[1].tolist()},
            "pro": {"x": pro[0].tolist(), "y": pro[1].tolist()},
        }
    else:
        results = {
            "auroc": float(auroc),
            "roc": {"x": roc[0].tolist(), "y": roc[1].tolist()},
        }
    with open(savepath, "w") as f:
        f.write(json.dumps(results, indent=2))
    return
