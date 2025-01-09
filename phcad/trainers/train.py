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
    device = "cuda" if (not device and torch.cuda.is_available) else "cpu"
    net.to(device)

    savepath = savedir / f"{savename}.pt"
    if savepath.exists():
        checkpoint = torch.load(savepath, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["model_state"])
        if len(checkpoint["epoch-loss"]) >= epochs:
            logger.info(
                f"Training already completed, returning saved model from {savepath}"
            )
            net.to("cpu").eval()
            return net
        opt.load_state_dict(checkpoint["opt_state"])
        sched = checkpoint["scheduler"]
        sched.optimizer = opt
        last_epoch = checkpoint["epoch-loss"][-1][0]
        logger.info(
            f"Resuming training from checkpoint {savepath} at epoch {last_epoch + 1}"
        )
    else:
        logger.info(f"Training started. Checkpoint path: {savepath}")
        last_epoch = 0
        checkpoint = {"epoch-loss": []}

    loss_function.to(device)
    net.train()
    for epoch in range(last_epoch + 1, epochs + 1):
        n_samps, total_loss = 0, 0
        for n_batch, data in enumerate(dloader):
            inputs, labels = data
            with torch.device(device):
                inputs = inputs.to(device)
                labels = labels.to(device)
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
    net.to("cpu").eval()
    return net
