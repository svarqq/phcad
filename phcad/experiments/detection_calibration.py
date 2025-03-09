import copy

import torch
from torch.utils.data import Subset, ConcatDataset
import torch.nn.functional as F

from phcad.utils import dsvdd_center
from phcad.experiments.constants import EXPROOT, NUMSEEDS
from phcad.models.constants import MODEL_MAP
from phcad.models.layers import PlattCal, BetaCal
from phcad.data.utils import (
    get_dataset,
    get_train_cal_splits,
    mean_std,
    BalancedLoader,
    DATASET_MAP,
)
from phcad.data.spectral_natural_images import SpectralNaturalImages
from phcad.data.transforms import (
    TEST_TRANSFORM_MAP,
    generic_norm_transform,
    label_to_zero,
    label_to_one,
    mask_to_class,
)
from phcad.data.constants import OE_DATASET_MAP
from phcad.train.losses import LOSS_MAP
from phcad.test.calibration_curves import calibration_curve


def get_calibration_curves(
    dataset_name,
    label,
    loss_name,
    spectral_oe_train=False,
    spectral_oe_cal=False,
    device=None,
):
    if loss_name not in LOSS_MAP:
        raise ValueError(f"loss_name must be one of {list(LOSS_MAP.keys())}")

    base_loss = LOSS_MAP[loss_name]
    base_loss_unsupervised = loss_name == "dsvdd" or loss_name == "ssim"
    if base_loss_unsupervised and spectral_oe_train:
        raise ValueError(
            f"spectral_oe_train can't be true for unsupervised loss {loss_name}"
        )

    transform_and_model_identifier = dataset_name
    if dataset_name != "fmnist" and dataset_name != "cifar10" and loss_name == "ssim":
        transform_and_model_identifier += "-ae"

    if loss_name == "ssim":
        if dataset_name == "cifar10" or dataset_name == "fmnist":
            base_loss = base_loss(win_size=11)
        else:
            base_loss = base_loss(win_size=11)

    # Setup save directories
    exp_dir = EXPROOT / "detection" / dataset_name / loss_name
    model_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "cal_curves"
    train_cal_splits_dir = exp_dir / "train-cal-splits"
    for p in [model_dir, results_dir, train_cal_splits_dir]:
        if not p.exists():
            p.mkdir(parents=True)

    base_model_args = {}
    if loss_name == "ssim":
        # Autoencoder
        base_model_args["ae"] = True
    elif loss_name == "bce":
        # Close open layer with FC to 1 node
        base_model_args["clf"] = True
    elif loss_name == "dsvdd":
        base_model_args["bias"] = False

    # Set up data
    _, _, dataset_type = DATASET_MAP[dataset_name]

    train_full = get_dataset(dataset_name, "train", label)
    mean_full, std_full = mean_std(train_full, ae=loss_name == "ssim")
    test_transform_full = TEST_TRANSFORM_MAP[transform_and_model_identifier](
        mean_full,
        std_full,
        ae=loss_name == "ssim",
        gs=dataset_name == "fmnist",
        resize_px=28 if dataset_name == "fmnist" else None,
    )

    if dataset_type == "classification":
        # detection
        indist_test_data = get_dataset(dataset_name, "test", label)
        indist_test_data.dataset.target_transform = label_to_zero
    elif dataset_type == "localization":
        # localized, subtle anomalies
        indist_test_data = get_dataset(
            dataset_name, "test", label, test_indist_only=True
        )
        indist_test_data.dataset.target_transform = mask_to_class
    indist_test_data.dataset.transform = test_transform_full

    oe_data_full = None
    if not (spectral_oe_train and spectral_oe_cal):
        oe_name = OE_DATASET_MAP[dataset_name]
        if oe_name == "cifar100":
            oe_train_1 = get_dataset(OE_DATASET_MAP[dataset_name], "train")
            oe_train_2 = get_dataset(OE_DATASET_MAP[dataset_name], "test")
            for oe_dataset in [oe_train_1, oe_train_2]:
                oe_dataset.dataset.transform = test_transform_full
                oe_dataset.dataset.target_transform = label_to_one
            oe_data_full = oe_train_1 + oe_train_2
        elif oe_name == "imagenet21k-minus1k":
            oe_data_full = get_dataset(oe_name, "train")
            oe_data_full.dataset.transform = test_transform_full
            oe_data_full.dataset.target_transform = label_to_one

    spectral_data = None
    if spectral_oe_train or spectral_oe_cal:
        imshape = indist_test_data[0][0].shape
        spectral_data = SpectralNaturalImages(
            imshape,
            transform=generic_norm_transform(mean_full, std_full),
            target=1,
        )

    if spectral_oe_cal:
        test_loader = BalancedLoader(indist_test_data, spectral_data)
    else:
        test_loader = BalancedLoader(indist_test_data, oe_data_full)

    train_copy = copy.deepcopy(train_full)
    oe_copy = copy.deepcopy(oe_data_full) if oe_data_full else None
    spectral_copy = copy.deepcopy(spectral_data) if spectral_data else None
    test_indist_copy = copy.deepcopy(indist_test_data)

    oe_train_type = ""
    if not base_loss_unsupervised and spectral_oe_train:
        oe_train_type = "spectral"
    elif not base_loss_unsupervised and not spectral_oe_train:
        oe_train_type = "oe"
    else:
        oe_train_type = "none"
    oe_cal_type = ""
    if spectral_oe_cal:
        oe_cal_type = "spectral"
    else:
        oe_cal_type = "oe"

    for seednum in range(NUMSEEDS):
        basename = f"{label}-{seednum}"

        # Prepare partial train, cal data with partial mean, std norm
        train_data_partial, _ = get_train_cal_splits(
            train_copy, idcs_savepath=train_cal_splits_dir / f"{basename}-train.json"
        )
        mean_partial, std_partial = mean_std(train_data_partial, ae=loss_name == "ssim")
        test_transform_partial = TEST_TRANSFORM_MAP[transform_and_model_identifier](
            mean_partial,
            std_partial,
            ae=loss_name == "ssim",
            gs=dataset_name == "fmnist",
            resize_px=28 if dataset_name == "fmnist" else None,
        )

        if spectral_oe_cal:
            spectral_copy.transform = generic_norm_transform(mean_partial, std_partial)
        else:
            if isinstance(oe_copy, ConcatDataset):
                for subset in oe_copy.datasets:
                    subset.dataset.transform = test_transform_partial
            elif isinstance(oe_copy, Subset):
                oe_copy.dataset.transform = test_transform_partial

        # Prepare test data with partial mean, std norm
        test_indist_copy.dataset.transform = test_transform_partial

        test_loader_ph = None
        if spectral_oe_cal:
            test_loader_ph = BalancedLoader(test_indist_copy, spectral_copy)
        else:
            test_loader_ph = BalancedLoader(test_indist_copy, oe_copy)

        # --------- Cal curves

        partial_pre = f"{basename}-partial-{oe_train_type}"

        # --------- Phtrain

        #  Load phtrain
        phtrain_pre = f"{partial_pre}-phtrain-{oe_cal_type}"
        model_phtrain = MODEL_MAP[transform_and_model_identifier](**base_model_args)
        model_phtrain.prepare_calibration_network()
        model_state = torch.load(model_dir / f"{phtrain_pre}.pt", weights_only=False)[
            "model_state"
        ]
        model_phtrain.load_state_dict(model_state)

        # Curve for phtrain
        results_path = results_dir / f"{phtrain_pre}.json"
        inputs_to_indist_pests_phtrain = lambda inputs: 1 - F.sigmoid(
            model_phtrain(inputs)
        )
        calibration_curve(
            inputs_to_indist_pests_phtrain,
            test_loader_ph,
            modules=[model_phtrain],
            savepath=results_path,
        )

        # -------

        # Load partial

        model_partial = MODEL_MAP[transform_and_model_identifier](**base_model_args)
        model_state = torch.load(model_dir / f"{partial_pre}.pt", weights_only=False)[
            "model_state"
        ]
        model_partial.load_state_dict(model_state)

        # Get logits and probability estimate functions
        if loss_name == "dsvdd":
            center_savepath = model_dir / f"{partial_pre}-center.pt"
            center = dsvdd_center(None, None, center_savepath)
            loss_partial = base_loss(center=center)
        else:
            loss_partial = base_loss
        logits_fn_partial, pests_fn_partial = (
            loss_partial.get_logits,
            loss_partial.get_pests,
        )
        modules_phcal = [model_partial, loss_partial]

        # --------- Platt

        # Load Platt
        platt_pre = f"{partial_pre}-platt-{oe_cal_type}"
        platt_state = torch.load(model_dir / f"{platt_pre}.pt", weights_only=False)
        pm = PlattCal()
        pm.load_state_dict(platt_state)
        modules_platt = modules_phcal + [pm]

        # Curve for Platt
        results_path = results_dir / f"{platt_pre}.json"
        inputs_to_indist_pests_platt = lambda inputs: 1 - F.sigmoid(
            pm(
                logits_fn_partial(
                    model_inputs=inputs, model_outputs=model_partial(inputs)
                )
            )
        )
        calibration_curve(
            inputs_to_indist_pests_platt,
            test_loader_ph,
            modules=modules_platt,
            savepath=results_path,
        )

        # --------- Beta

        # Load Beta
        beta_pre = f"{partial_pre}-beta-{oe_cal_type}"
        beta_state = torch.load(model_dir / f"{beta_pre}.pt", weights_only=False)
        bm = BetaCal()
        bm.load_state_dict(beta_state)
        modules_beta = modules_phcal + [bm]

        # Curve for Beta
        results_path = results_dir / f"{beta_pre}.json"
        inputs_to_indist_pests_beta = lambda inputs: 1 - F.sigmoid(
            bm(
                pests_fn_partial(
                    model_inputs=inputs, model_outputs=model_partial(inputs)
                )
            )
        )  # Note bm outputs logits
        calibration_curve(
            inputs_to_indist_pests_beta,
            test_loader_ph,
            modules=modules_beta,
            savepath=results_path,
        )

        # --------- Full
        full_pre = f"{basename}-full-{oe_train_type}"

        # Get probability estimates function
        if loss_name == "dsvdd":
            center_savepath = model_dir / f"{full_pre}-center.pt"
            center = dsvdd_center(None, None, center_savepath)
            loss_full = base_loss(center=center)
        else:
            loss_full = base_loss
        pests_fn_full = loss_full.get_pests

        # Load full
        model_full = MODEL_MAP[transform_and_model_identifier](**base_model_args)
        model_state = torch.load(model_dir / f"{full_pre}.pt", weights_only=False)[
            "model_state"
        ]
        model_full.load_state_dict(model_state)
        modules_full = [model_full, loss_full]

        # Curve for full
        results_path = results_dir / f"{full_pre}.json"
        inputs_to_indist_pests_full = lambda inputs: 1 - pests_fn_full(
            model_inputs=inputs, model_outputs=model_full(inputs)
        )
        calibration_curve(
            inputs_to_indist_pests_full,
            test_loader,
            modules=modules_full,
            savepath=results_path,
        )
