import logging

import torch

from phcad.train.constants import CHKPTDIR

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
    device = "cuda" if (not device and torch.cuda.is_available()) else "cpu"
    net.to(device)

    savepath = savedir / f"{savename}.pt"
    if savepath.exists():
        try:
            checkpoint = torch.load(savepath, map_location="cpu", weights_only=False)
            if len(checkpoint["epoch-loss"]) >= epochs:
                net.load_state_dict(checkpoint["model_state"])
                logger.info(
                    f"Training already completed, returning saved model from {savepath}"
                )
                net.to("cpu").eval()
                return net

            all_states_present = (
                "model_state" in checkpoint
                and "opt_state" in checkpoint
                and "sched_state" in checkpoint
            )
            assert all_states_present

            net.load_state_dict(checkpoint["model_state"])
            opt.load_state_dict(checkpoint["opt_state"])
            sched.load_state_dict(checkpoint["sched_state"])
            sched.optimizer = opt
            last_epoch = checkpoint["epoch-loss"][-1][0]
            logger.info(
                f"Resuming training from checkpoint {savepath} at epoch {last_epoch + 1}"
            )
        except Exception:
            logger.info(
                f"Failed to load checkpoint at {savepath}, rerunning training process"
            )
            last_epoch = 0
            checkpoint = {"epoch-loss": []}
    else:
        logger.info(f"Training started. Checkpoint path: {savepath}")
        last_epoch = 0
        checkpoint = {"epoch-loss": []}

    loss_function.to(device)
    net.train()
    for epoch in range(last_epoch + 1, epochs + 1):
        n_samps, total_loss = 0, 0
        for _, data in enumerate(dloader):
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
        checkpoint["sched_state"] = sched.state_dict()
        torch.save(
            checkpoint,
            savepath,
        )
        logger.info(f"Completed epoch {epoch}")
    net.to("cpu").eval()
    return net
