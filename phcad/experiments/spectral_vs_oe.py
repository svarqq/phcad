import torch
import torchvision.transforms.v2.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from phcad.experiments.constants import EXPDIR
from phcad.models.constants import MODEL_MAP
from phcad.data_handling.utils import get_dataset, mean_std, BalancedLoader
from phcad.data_handling.spectral_natural_images import SpectralNaturalImages
from phcad.data_handling.transforms import (
    TRAIN_TRANSFORM_MAP,
    generic_norm_transform,
    indist_target_transform,
    anom_target_transform,
)
from phcad.data_handling.constants import OE_DATASET_MAP
from phcad.trainers.losses import CompositeBCE
from phcad.trainers.train import train
from phcad.test.anomaly_scores import probability_estimates
from phcad.test.evaluate import evaluate_thresholding
from phcad.trainers.utils import get_optim_sched_epochs


loss = CompositeBCE()
anomaly_score_function = probability_estimates


def run_spectral_vs_oe(dataset_name, label, device=None):
    exp_dir = EXPDIR / "spectral-v-oe" / dataset_name
    model_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
    for p in [model_dir, results_dir]:
        if not p.exists():
            p.mkdir(parents=True)

    train_data = get_dataset(dataset_name, "train", label)
    mean, std = mean_std(train_data)
    transform = TRAIN_TRANSFORM_MAP[dataset_name](mean, std)
    train_data.dataset.transform = transform
    train_data.dataset.target_transform = indist_target_transform

    test_data_in = get_dataset(dataset_name, "test", label)
    test_data_in.dataset.transform = generic_norm_transform(mean, std)
    test_data_in.dataset.target_transform = indist_target_transform
    test_data_anom = get_dataset(dataset_name, "test", label, complement=True)
    test_data_anom.dataset.transform = generic_norm_transform(mean, std)
    test_data_anom.dataset.target_transform = anom_target_transform
    test_data = test_data_in + test_data_anom

    oe_data_1 = get_dataset(OE_DATASET_MAP[dataset_name], "train")
    oe_data_2 = get_dataset(OE_DATASET_MAP[dataset_name], "test")
    for dataset in [oe_data_1, oe_data_2]:
        dataset.dataset.transform = transform
        dataset.dataset.target_transform = anom_target_transform
    oe_data = oe_data_1 + oe_data_2

    imshape = train_data[0][0].shape
    spectral_data = SpectralNaturalImages(
        imshape,
        transform=generic_norm_transform(mean, std),
        target=1,
    )

    oe_loader = BalancedLoader(train_data, oe_data)
    spectral_loader = BalancedLoader(train_data, spectral_data)
    test_loader = DataLoader(test_data, 128, num_workers=4)

    for seed in range(5):
        basename = f"oe-{label}-{seed}"
        results_path = results_dir / f"{basename}.json"
        model = MODEL_MAP[dataset_name](clf=True)
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model.parameters())
        sched = sched(opt)
        model = train(
            epochs=epochs,
            net=model,
            loss_function=loss,
            opt=opt,
            sched=sched,
            dloader=oe_loader,
            savename=basename,
            savedir=model_dir,
        )
        evaluate_thresholding(model, test_loader, anomaly_score_function, results_path)

        basename = f"spectral-{label}-{seed}"
        results_path = results_dir / f"{basename}.json"
        model = MODEL_MAP[dataset_name](clf=True)
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model.parameters())
        sched = sched(opt)
        model = train(
            epochs=epochs,
            net=model,
            loss_function=loss,
            opt=opt,
            sched=sched,
            dloader=spectral_loader,
            savename=basename,
            savedir=model_dir,
        )
        evaluate_thresholding(model, test_loader, anomaly_score_function, results_path)
