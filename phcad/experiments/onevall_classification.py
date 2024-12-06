import torch

from phcad.models.ae_mvtec import AEMvTec
from phcad.data_handling.mvtec_dataset import mvtec_train_cal_dataloaders
from phcad.trainers.train import train
from phcad.trainers.losses import SSIMLoss


def run_onevall_exp_mvtec_ae(device=None):
    torch.set_default_dtype(torch.double)
    labels = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    resize_px = 276
    crop_px = 256
    loss_function = SSIMLoss()

    for seed in range(1, 6):
        print(f"On seed {seed}")
        for label in labels:
            print(f"On label {label}")

            print("Training full")
            ae = AEMvTec()
            savename = f"mvtec-ae-full-{label}-{seed}"
            train_size = 10000
            cal_size = 0
            trainloader, _ = mvtec_train_cal_dataloaders(
                label, resize_px, crop_px, train_size, cal_size
            )
            opt = torch.optim.Adam(ae.parameters())
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
            train(
                50, ae, loss_function, opt, sched, trainloader, savename, device=device
            )

            print("Training partial")
            ae = AEMvTec()
            savename = f"mvtec-ae-partial-{label}-{seed}"
            train_size = 7500
            cal_size = 2500
            trainloader, _ = mvtec_train_cal_dataloaders(
                label, resize_px, crop_px, train_size, cal_size
            )
            opt = torch.optim.Adam(ae.parameters())
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=(20, 40))
            train(
                50, ae, loss_function, opt, sched, trainloader, savename, device=device
            )
