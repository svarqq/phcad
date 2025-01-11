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

from phcad.data_handling.transforms import mask_to_class
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


def evaluate_thresholding_segmentation(
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
    anomaly_scores, targets = [], []
    for data, batch_target_masks in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)
            batch_maps = inputs_to_anomaly_map(data)
            rsz_pred_maps = F.resize(batch_maps, batch_target_masks[-2:].shape)
        anomaly_scores.append(rsz_pred_maps.to("cpu").detach())
        targets.append(batch_target_masks.to("cpu").detach())
    for module in modules:
        module.to("cpu")

    anomaly_scores = torch.cat((*anomaly_scores,))
    targets = torch.cat((*targets,))

    auroc = roc_auc_score(targets.numpy().view(-1), anomaly_scores.numpy().view(-1))
    roc = roc_curve(targets.numpy().view(-1), anomaly_scores.numpy().view(-1))[:2]
    if gen_aupro:
        aupro_obj = AUPRO()
        aupro_obj.update(preds=anomaly_scores, target=targets.to(torch.uint8))
        pro = aupro_obj._compute()
        aupro = aupro_obj.compute()

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


def evaluate_thresholding_segmentation_perturbation(
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
    anomaly_scores, targets = [], []
    for data, batch_target_masks in test_loader:
        with torch.device(device), torch.no_grad():
            data = data.to(device)

            rsz_masks = F.resize(batch_target_masks, data.shape[-2:])
            rsz_masks[rsz_masks >= 0.5] = 1
            rsz_masks[rsz_masks < 0.5] = 0
            if detection_targets_for_loss:
                grad_targets = torch.stack(list(map(mask_to_class, batch_target_masks)))
            else:
                grad_targets = rsz_masks
            targets = targets.to(device)

            with torch.enable_grad():
                data.requires_grad = True
                loss = inputs_to_loss(data, grad_targets)
                loss.backward()
            signs = torch.ge(data.grad, 0) * 2 - 1
            perturbations = norm(signs * eps)

            perturbed_data = data - perturbations
            batch_maps = inputs_to_anomaly_map(perturbed_data)
            rsz_pred_maps = F.resize(batch_maps, batch_target_masks.shape[-2:])
        anomaly_scores.append(rsz_pred_maps.to("cpu").detach())
        targets.append(batch_target_masks.to("cpu").detach())
    for module in modules:
        module.to("cpu")

    anomaly_scores = torch.cat((*anomaly_scores,))
    targets = torch.cat((*targets,))

    auroc = roc_auc_score(targets.numpy().view(-1), anomaly_scores.numpy().view(-1))
    roc = roc_curve(targets.numpy().view(-1), anomaly_scores.numpy().view(-1))[:2]
    if gen_aupro:
        aupro_obj = AUPRO()
        aupro_obj.update(preds=anomaly_scores, target=targets.to(torch.uint8))
        pro = aupro_obj._compute()
        aupro = aupro_obj.compute()

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
