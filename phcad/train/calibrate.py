import logging

import torch
from torch.optim import LBFGS
import torch.nn.functional as F

from phcad.models.layers import PlattCal, PerPixelPlatt, BetaCal, PerPixelBeta

logger = logging.getLogger(__name__)


def apply_posthoc_calibration(
    cal_type: str,  # platt or beta
    inputs_to_val_to_cal,  # logits for platt, probability estimates for beta
    dataloader,
    savepath=None,
    num_scores=20000,
    modules_in_fn=[],
    device=None,
):
    if cal_type == "platt":
        cal_module = PlattCal()
    elif cal_type == "beta":
        cal_module = BetaCal()
    else:
        raise ValueError('Argument for cal_type must be one of ["platt", "beta"]')
    if savepath and savepath.exists():
        logger.info(f"Platt scaling already completed, loading state from {savepath}")
        pm_state = torch.load(savepath, map_location="cpu", weights_only=True)
        cal_module.load_state_dict(pm_state)
        return cal_module

    logger.info(f"{cal_type.capitalize()} calibrating, saving to {savepath}")
    device = "cuda" if not device and torch.cuda.is_available else "cpu"

    for module in modules_in_fn:
        module.eval()
        module.to(device)
    vals_to_cal, targets = (
        torch.empty(num_scores).to(device),
        torch.empty(num_scores).to(device),
    )
    ct = 0

    while ct < num_scores:
        for data, batch_targets in dataloader:
            with torch.device(device), torch.no_grad():
                data = data.to(device)
                batch_vals_to_cal = inputs_to_val_to_cal(data)

                if ct + len(batch_vals_to_cal) <= num_scores:
                    vals_to_cal[ct : ct + len(batch_vals_to_cal)] = batch_vals_to_cal
                    targets[ct : ct + len(batch_vals_to_cal)] = batch_targets
                else:
                    vals_to_cal[ct:] = batch_vals_to_cal[: num_scores - ct]
                    targets[ct:] = targets[: num_scores - ct]

                ct += len(batch_vals_to_cal)
                if ct >= num_scores:
                    break
    for module in modules_in_fn:
        module.to("cpu")

    loss_fn = F.binary_cross_entropy_with_logits
    cal_module.train()
    cal_module.to(device)
    optim = LBFGS(cal_module.parameters(), max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        new_estimates = cal_module(vals_to_cal)
        loss = loss_fn(new_estimates, targets)
        loss.backward()
        return loss

    optim.step(closure)
    cal_module.eval()
    cal_module.to("cpu")

    if savepath:
        torch.save(cal_module.state_dict(), savepath)
    return cal_module


def apply_posthoc_calibration_seg(
    cal_type,
    inputs_to_val_to_cal,  # logits for platt, probability estimates for beta
    dataloader,
    epochs,
    opt,
    sched,
    wh_shape,
    modules_in_fn=[],
    savepath=None,
    device=None,
):
    if cal_type == "platt":
        cal_module = PerPixelPlatt(wh_shape)
    elif cal_type == "beta":
        cal_module = PerPixelBeta(wh_shape)
    else:
        raise ValueError('Argument for cal_type must be one of ["platt", "beta"]')
    device = "cuda" if (not device and torch.cuda.is_available) else "cpu"
    cal_module.to(device)
    opt = opt(cal_module.parameters())
    sched = sched(opt)

    if savepath.exists():
        try:
            checkpoint = torch.load(savepath, map_location="cpu", weights_only=False)
            if len(checkpoint["epoch-loss"]) >= epochs:
                cal_module.load_state_dict(checkpoint["model_state"])
                logger.info(
                    f"Training already completed, returning saved model from {savepath}"
                )
                cal_module.to("cpu").eval()
                return cal_module

            all_states_present = (
                "model_state" in checkpoint
                and "opt_state" in checkpoint
                and "sched_state" in checkpoint
            )
            assert all_states_present

            cal_module.load_state_dict(checkpoint["model_state"])
            opt.load_state_dict(checkpoint["opt_state"])
            sched.load_state_dict(checkpoint["sched_state"])
            last_epoch = checkpoint["epoch-loss"][-1][0]
            logger.info(
                f"Resuming training from checkpoint {savepath} at epoch {last_epoch + 1}"
            )
        except Exception:
            logger.info(
                f"Failed to load checkpoint at {savepath}, rerunning calibration process"
            )
            last_epoch = 0
            checkpoint = {"epoch-loss": []}
    else:
        logger.info(f"Training started. Checkpoint path: {savepath}")
        last_epoch = 0
        checkpoint = {"epoch-loss": []}

    for module in modules_in_fn:
        module.eval()
        module.to(device)
    cal_module.train()
    for epoch in range(last_epoch + 1, epochs + 1):
        n_samps, total_loss = 0, 0
        for _, data in enumerate(dataloader):
            imgs, target_masks = data
            with torch.device(device), torch.no_grad():
                imgs = imgs.to(device)
                target_masks = target_masks.to(device)
                opt.zero_grad()
                vals_to_cal = inputs_to_val_to_cal(imgs)
                with torch.enable_grad():
                    new_estimates = cal_module(vals_to_cal)
                    loss = F.binary_cross_entropy_with_logits(
                        new_estimates, target_masks
                    )
                    loss.backward()
                    opt.step()

            n_samps += len(imgs)
            total_loss += loss.item() * len(imgs)
        sched.step()

        epoch_loss = total_loss / n_samps
        checkpoint["epoch-loss"].append((epoch, epoch_loss))
        checkpoint["model_state"] = cal_module.state_dict()
        checkpoint["opt_state"] = opt.state_dict()
        checkpoint["sched_state"] = sched.state_dict()
        torch.save(
            checkpoint,
            savepath,
        )
        logger.info(f"Completed epoch {epoch}")

    for module in modules_in_fn:
        module.to("cpu")
    cal_module.to("cpu").eval()
    return cal_module
