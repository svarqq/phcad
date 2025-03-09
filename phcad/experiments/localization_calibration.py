import copy

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from phcad.experiments.constants import EXPROOT, NUMSEEDS
from phcad.models.constants import SEG_MODEL_MAP
from phcad.models.fcdd import ReceptiveUpsample
from phcad.models.layers import PerPixelPlatt, PerPixelBeta
from phcad.data.utils import (
    get_dataset,
    get_train_cal_splits,
    mean_std,
    BalancedLoader,
    DATASET_MAP,
)
from phcad.data.spectral_natural_images import SpectralNaturalImages
from phcad.data.transforms import (
    SEG_TEST_TRANSFORM_MAP,
    generic_norm_transform,
    synthetic_mask,
)
from phcad.data.constants import OE_DATASET_MAP
from phcad.train.losses import SEG_LOSS_MAP
from phcad.test.calibration_curves import calibration_curve


def get_seg_cal_curves(
    dataset_name,
    label,
    loss_name,
    spectral_oe_train=False,
    spectral_oe_cal=False,
    device=None,
):
    if loss_name not in SEG_LOSS_MAP:
        raise ValueError(f"loss_name must be one of {list(SEG_LOSS_MAP.keys())}")

    base_loss = SEG_LOSS_MAP[loss_name]
    base_loss_unsupervised = loss_name == "dsvdd" or loss_name == "ssim"
    if base_loss_unsupervised and spectral_oe_train:
        raise ValueError(
            f"spectral_oe_train can't be true for unsupervised loss {loss_name}"
        )

    if loss_name == "ssim":
        base_loss = base_loss(reduce=False, win_size=11, pad=True)
    elif loss_name == "fcdd":
        receptive_upsample = ReceptiveUpsample()
        base_loss = base_loss(receptive_upsample)

    # Setup save directories
    exp_dir = EXPROOT / "localization" / dataset_name / loss_name
    model_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "cal_curves"
    train_cal_splits_dir = exp_dir / "train-cal-splits"
    for p in [model_dir, results_dir, train_cal_splits_dir]:
        if not p.exists():
            p.mkdir(parents=True)

    # Set up data
    _, _, dataset_type = DATASET_MAP[dataset_name]

    train_full = get_dataset(dataset_name, "train", label)
    mean_full, std_full = mean_std(train_full, ae=loss_name == "ssim")
    test_transform_full = SEG_TEST_TRANSFORM_MAP[loss_name](
        mean_full, std_full, ae=loss_name == "ssim"
    )

    indist_test_data = get_dataset(dataset_name, "test", label, test_indist_only=True)
    indist_test_data.dataset.transform = test_transform_full
    imshape = indist_test_data[0][0].shape
    indist_test_data.dataset.target_transform = lambda im: v2.functional.resize(
        im.unsqueeze(0), imshape[-2:]
    ).squeeze(0)

    oe_data_full = None
    if not (spectral_oe_train and spectral_oe_cal):
        oe_name = OE_DATASET_MAP[dataset_name]
        oe_data_full = get_dataset(oe_name, "train")
        oe_data_full.dataset.transform = test_transform_full
        oe_data_full.dataset.target_transform = synthetic_mask(
            imshape[-2:], anomaly_targets=True
        )

    spectral_data = None
    if spectral_oe_train or spectral_oe_cal:
        spectral_data = SpectralNaturalImages(
            imshape,
            transform=generic_norm_transform(mean_full, std_full),
            localization_targets=True,
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

    for seed in range(NUMSEEDS):
        basename = f"{label}-{seed}"

        # Prepare partial train, cal data with partial mean, std norm
        train_data_partial, _ = get_train_cal_splits(
            train_copy, idcs_savepath=train_cal_splits_dir / f"{basename}-train.json"
        )
        mean_partial, std_partial = mean_std(train_data_partial, ae=loss_name == "ssim")
        test_transform_partial = SEG_TEST_TRANSFORM_MAP[loss_name](
            mean_partial, std_partial, ae=loss_name == "ssim"
        )

        if spectral_oe_cal:
            spectral_copy.transform = generic_norm_transform(mean_partial, std_partial)
        else:
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

        # Load partial

        model_partial = SEG_MODEL_MAP[loss_name]
        model_state = torch.load(model_dir / f"{partial_pre}.pt", weights_only=False)[
            "model_state"
        ]
        model_partial.load_state_dict(model_state)

        # Get logits and probability estimate functions
        logits_fn_partial, pests_fn_partial = (
            base_loss.get_logits,
            base_loss.get_pests,
        )
        modules_phcal = [model_partial, base_loss]

        # --------- Platt

        # Load Platt
        platt_pre = f"{partial_pre}-platt-{oe_cal_type}"
        platt_state = torch.load(model_dir / f"{platt_pre}.pt", weights_only=False)[
            "model_state"
        ]
        pm = PerPixelPlatt(imshape[-2:])
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
            localization=True,
            modules=modules_platt,
            savepath=results_path,
        )

        # --------- Beta

        # Load Beta
        beta_pre = f"{partial_pre}-beta-{oe_cal_type}"
        beta_state = torch.load(model_dir / f"{beta_pre}.pt", weights_only=False)[
            "model_state"
        ]
        bm = PerPixelBeta(imshape[-2:])
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
            localization=True,
            modules=modules_beta,
            savepath=results_path,
        )

        # --------- Full
        full_pre = f"{basename}-full-{oe_train_type}"

        # Get probability estimates function
        pests_fn_full = base_loss.get_pests

        # Load full
        model_full = SEG_MODEL_MAP[loss_name]
        model_state = torch.load(model_dir / f"{full_pre}.pt", weights_only=False)[
            "model_state"
        ]
        model_full.load_state_dict(model_state)
        modules_full = [model_full, base_loss]

        # Curve for full
        results_path = results_dir / f"{full_pre}.json"
        inputs_to_indist_pests_full = lambda inputs: 1 - pests_fn_full(
            model_inputs=inputs, model_outputs=model_full(inputs)
        )
        calibration_curve(
            inputs_to_indist_pests_full,
            test_loader,
            localization=True,
            modules=modules_full,
            savepath=results_path,
        )
