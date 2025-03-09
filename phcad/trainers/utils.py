from functools import partial

import torch


def get_optim_sched_epochs(dataset):
    opt = partial(torch.optim.Adam, lr=1e-3)
    sched, epochs = None, 0
    if dataset == "fmnist" or dataset == "cifar10":
        sched = partial(
            torch.optim.lr_scheduler.MultiStepLR, milestones=[100, 150], gamma=0.1
        )
        epochs = 200
    elif dataset == "mpdd" or dataset == "mvtec":
        sched = partial(
            torch.optim.lr_scheduler.MultiStepLR, milestones=[20, 25], gamma=0.1
        )
        epochs = 30  # Really 300, because of dataset extension
    return opt, sched, epochs
