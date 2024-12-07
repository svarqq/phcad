import torch
import logging
import itertools

from phcad.models.ae_mvtec import AEMvTec
from phcad.data_handling.mvtec_dataset import mvtec_train_cal_dataloaders
from phcad.trainers.train import train
from phcad.trainers.losses import SSIMLoss, CompositeBCE
from phcad.trainers.constants import CHKPTDIR


def run_onevall_exp_mvtec_ae(label, device=None):
    torch.set_default_dtype(torch.double)
    resize_px = 276
    crop_px = 256
    loss_function = SSIMLoss()
    weight_decay = 1e-5

    train_size = 10000
    cal_size = 0
    tl_full, _ = mvtec_train_cal_dataloaders(
        label, resize_px, crop_px, train_size, cal_size
    )
    train_size = 7500
    cal_size = 2500
    tl_partial, _ = mvtec_train_cal_dataloaders(
        label, resize_px, crop_px, train_size, cal_size
    )
    for seed in range(1, 6):
        logging.info(f"On seed {seed}")

        logging.info("Training full")
        ae = AEMvTec()
        savename = f"mvtec-ae-full-{label}-{seed}"
        opt = torch.optim.Adam(ae.parameters(), weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
        train(50, ae, loss_function, opt, sched, tl_full, savename, device=device)

        logging.info("Training partial")
        ae = AEMvTec()
        savename = f"mvtec-ae-partial-{label}-{seed}"
        opt = torch.optim.Adam(ae.parameters(), weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
        train(50, ae, loss_function, opt, sched, tl_partial, savename, device=device)


def calibrate_mvtec_ae(label, device=None):
    torch.set_default_dtype(torch.double)
    resize_px = 276
    crop_px = 256
    loss_function = CompositeBCE()
    weight_decay = 1e-5

    train_size = 7500
    cal_size = 2500
    _, cal_loader = mvtec_train_cal_dataloaders(
        label, resize_px, crop_px, train_size, cal_size
    )
    for seed in range(1, 6):
        loaddir = CHKPTDIR / f"mvtec-ae-partial-{label}-{seed}.pt"
        logging.info(f"On seed {seed}")
        for nlayers in range(1, 4):
            savename = f"mvtec-ae-cal-{label}-nl{nlayers}-{seed}"
            logging.info(f"Calibrating, holding {nlayers} frozen")
            ae = AEMvTec()
            try:
                state = torch.load(loaddir)
            except FileNotFoundError:
                logging.warn(f"{loaddir} does not exist yet -- needs to be trained!")
                continue
            if len(state["epoch-loss"]) != 50:
                logging.warn(f"{loaddir} not yet completed!")
                continue
            ae.load_state_dict(state["model_state"])
            head_layers = ae.setup_cal((crop_px, crop_px), nlayers)
            params = list(
                itertools.chain.from_iterable(
                    map(lambda x: x.parameters(), head_layers)
                )
            )
            opt = torch.optim.Adam(params, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
            train(
                50, ae, loss_function, opt, sched, cal_loader, savename, device=device
            )
