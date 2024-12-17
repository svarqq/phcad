from functools import partial

from torch import optim


def get_optim_sched_epochs(dataset):
    opt, sched = None, None
    if dataset == "fmnist" or dataset == "cifar10":
        opt = partial(optim.Adam, lr=1e-03)
        sched = partial(
            optim.lr_scheduler.MultiStepLR, milestones=[100, 150], gamma=0.1
        )
        epochs = 200
    return opt, sched, epochs
