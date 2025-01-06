import logging

import torch
from torch.optim import LBFGS
import torch.nn.functional as F

from phcad.models.layers import PlattCal, BetaCal

logger = logging.getLogger(__name__)


def apply_posthoc_calibration(
    cal_type: str,  # platt or beta
    inputs_to_val_to_cal,  # logits for platt, probability estimates for beta
    dataloader,
    savepath=None,
    num_scores=50000,
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
