import torch
import logging

from phcad.models.ae_mvtec import AEMvTec
from phcad.data_handling.mvtec_dataset import mvtec_train_cal_dataloaders
from phcad.trainers.train import train
from phcad.trainers.losses import SSIMLoss


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

        print("Training full")
        ae = AEMvTec()
        savename = f"mvtec-ae-full-{label}-{seed}"
        opt = torch.optim.Adam(ae.parameters(), weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
        train(50, ae, loss_function, opt, sched, tl_full, savename, device=device)

        print("Training partial")
        ae = AEMvTec()
        savename = f"mvtec-ae-partial-{label}-{seed}"
        opt = torch.optim.Adam(ae.parameters(), weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
        train(50, ae, loss_function, opt, sched, tl_partial, savename, device=device)
