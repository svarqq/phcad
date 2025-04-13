import logging

import torch

logger = logging.getLogger(__name__)


def dsvdd_center(model, trainloader, savepath=None, device=None):
    if savepath.exists():
        logger.info(f"Loading DSVDD center from {savepath}")
        return torch.load(savepath, weights_only=True)

    device = "cuda" if (not device and torch.cuda.is_available) else "cpu"
    model.to(device)
    model.eval()

    # Find output dimensions of model
    for inputs, _ in trainloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            sample_output = model(inputs[0].unsqueeze(0))
        break
    output_dims = sample_output.shape[-1]

    center, nsamps = torch.zeros(output_dims), 0
    eps = 1e-1
    for inputs, _ in trainloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            center += outputs.cpu().mean(0)
            nsamps += len(inputs)
    model.to("cpu")

    center = center / nsamps
    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps

    logger.info(f"Saving DSVDD center to {savepath}")
    torch.save(center, savepath)
    return center
