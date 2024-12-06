import logging

import torch

from phcad.trainers.constants import CHKPTDIR

logger = logging.getLogger(__name__)


def train(
    epochs,
    net,
    loss_function,
    opt,
    sched,
    dloader,
    savename,
    savedir=CHKPTDIR,
    device=None,
):
    savepath = savedir / f"{savename}.pt"
    logger.info(f"Training started. Checkpoint path: {savepath}")
    if not device and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    checkpoint = {"epoch-loss": []}
    loss_function = loss_function.to(device)
    net.to(device)
    net.train()
    for epoch in range(1, epochs + 1):
        n_samps, total_loss = 0, 0
        for n_batch, data in enumerate(dloader):
            inputs, labels = data
            with torch.device(device):
                inputs = inputs.to(device)
                opt.zero_grad()
                outputs = net(inputs)
                loss = loss_function(
                    model_inputs=inputs, model_outputs=outputs, labels=labels
                )
                loss.backward()
                opt.step()

            n_samps += len(inputs)
            total_loss += loss.item() * len(inputs)
        sched.step()

        epoch_loss = total_loss / n_samps
        checkpoint["epoch-loss"].append((epoch, epoch_loss))
        checkpoint["model_state"] = net.state_dict()
        checkpoint["opt_state"] = opt.state_dict()
        checkpoint["scheduler"] = sched
        torch.save(
            checkpoint,
            savepath,
        )
        logger.info(f"Completed epoch {epoch}")
        break
    return net
