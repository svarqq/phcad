import torch

from phcad.trainers.constants import CHKPTDIR


def train(
    epochs,
    net,
    loss_function,
    opt,
    sched,
    dloader,
    savename,
    savedir=CHKPTDIR,
    device="cpu",
):
    savepath = savedir / f"{savename}.pt"
    checkpoint = {"epoch-loss": [], "model_state": None, "opt_state": None}
    net.to(device)
    net.train()
    for epoch in range(epochs):
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
            total_loss += loss.item() * n_samps
        sched.step()

        epoch_loss = total_loss / n_samps
        checkpoint["epoch-loss"].append((epoch, epoch_loss))
        checkpoint["model_state"] = net.state_dict()
        checkpoint["opt_state"] = opt.state_dict()
        torch.save(
            checkpoint,
            savepath,
        )
    return net
