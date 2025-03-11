import copy

from torch.utils.data import DataLoader
import torch.nn.functional as F

from phcad.experiments.constants import EXPROOT, NUMSEEDS
from phcad.models.constants import SEG_MODEL_MAP
from phcad.models.fcdd import ReceptiveUpsample
from phcad.data.utils import (
    get_dataset,
    get_train_cal_splits,
    mean_std,
    BalancedLoader,
    DATASET_MAP,
)
from phcad.data.spectral_natural_images import SpectralNaturalImages
from phcad.data.transforms import (
    SEG_TRAIN_TRANSFORM_MAP,
    SEG_TEST_TRANSFORM_MAP,
    generic_norm_transform,
    label_to_one,
    label_to_zero,
    synthetic_mask,
)
from phcad.data.constants import OE_DATASET_MAP, MVTEC_LABELS_NOFLIP
from phcad.train.losses import SEG_LOSS_MAP
from phcad.train.train import train
from phcad.train.calibrate import apply_posthoc_calibration_seg
from phcad.test.anomaly_scores import SEG_ANOMALY_SCORES
from phcad.test.evaluate import (
    evaluate_thresholding_localization,
    evaluate_thresholding_localization_perturbation,
)
from phcad.train.utils import get_optim_sched_epochs


def run_localization_experiment(
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
    base_loss_unsupervised = loss_name == "ssim"
    if base_loss_unsupervised and spectral_oe_train:
        raise ValueError(
            f"spectral_oe_train can't be true for unsupervised loss {loss_name}"
        )

    anomaly_score = SEG_ANOMALY_SCORES[loss_name]
    if loss_name == "ssim":
        base_loss = base_loss(reduce=False, win_size=11, pad=True)
        anomaly_score = anomaly_score(reduce=False, win_size=11, pad=True)
    elif loss_name == "fcdd":
        receptive_upsample = ReceptiveUpsample()
        base_loss = base_loss(receptive_upsample)
        anomaly_score = anomaly_score(receptive_upsample)

    # Setup save directories
    exp_dir = EXPROOT / "localization" / dataset_name / loss_name
    model_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
    train_cal_splits_dir = exp_dir / "train-cal-splits"
    for p in [model_dir, results_dir, train_cal_splits_dir]:
        if not p.exists():
            p.mkdir(parents=True)

    # Set up data
    _, _, dataset_type = DATASET_MAP[dataset_name]

    train_full = get_dataset(dataset_name, "train", label)
    mean_full, std_full = mean_std(train_full, ae=loss_name == "ssim")
    flip = True
    if "mvtec" in dataset_name and label in MVTEC_LABELS_NOFLIP:
        flip = False
    train_transform_full = SEG_TRAIN_TRANSFORM_MAP[loss_name](
        mean_full, std_full, ae=loss_name == "ssim", flip=flip
    )
    test_transform_full = SEG_TEST_TRANSFORM_MAP[loss_name](
        mean_full, std_full, ae=loss_name == "ssim"
    )

    train_full.dataset.transform = train_transform_full
    imshape = train_full[0][0].shape
    train_full.dataset.target_transform = (
        label_to_zero if loss_name == "fcdd" else synthetic_mask(imshape[-2:])
    )

    oe_data_full = None
    if not (spectral_oe_train and spectral_oe_cal):
        oe_name = OE_DATASET_MAP[dataset_name]
        oe_data_full = get_dataset(oe_name, "train")
        oe_data_full.dataset.transform = train_transform_full
        oe_data_full.dataset.target_transform = (
            label_to_one
            if loss_name == "fcdd"
            else synthetic_mask(imshape[-2:], anomaly_targets=True)
        )

    spectral_data = None
    if spectral_oe_train or spectral_oe_cal:
        localization_targets = False if loss_name == "fcdd" else True
        spectral_data = SpectralNaturalImages(
            imshape,
            transform=generic_norm_transform(mean_full, std_full),
            localization_targets=localization_targets,
            target=1,
        )

    train_loader_full = None
    if base_loss_unsupervised:
        train_loader_full = DataLoader(train_full, 128, num_workers=4)
    else:
        if spectral_oe_train:
            train_loader_full = BalancedLoader(train_full, spectral_data)
        else:
            train_loader_full = BalancedLoader(train_full, oe_data_full)

    test_data = get_dataset(dataset_name, "test", label)
    test_data.dataset.transform = test_transform_full
    test_loader = DataLoader(test_data, 128, num_workers=4)

    train_copy = copy.deepcopy(train_full)
    oe_copy = copy.deepcopy(oe_data_full) if oe_data_full else None
    spectral_copy = copy.deepcopy(spectral_data) if spectral_data else None
    test_copy = copy.deepcopy(test_data)

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
        train_data_partial, cal_data_indist = get_train_cal_splits(
            train_copy, idcs_savepath=train_cal_splits_dir / f"{basename}-train.json"
        )
        mean_partial, std_partial = mean_std(train_data_partial, ae=loss_name == "ssim")
        train_transform_partial = SEG_TRAIN_TRANSFORM_MAP[loss_name](
            mean_partial, std_partial, ae=loss_name == "ssim", flip=flip
        )
        test_transform_partial = SEG_TEST_TRANSFORM_MAP[loss_name](
            mean_partial, std_partial, ae=loss_name == "ssim"
        )

        for ds in [train_data_partial, cal_data_indist]:
            ds.dataset.transform = train_transform_partial

        if loss_name == "fcdd":
            cal_data_indist = copy.deepcopy(cal_data_indist)
            cal_data_indist.dataset.target_transform = synthetic_mask(imshape[-2:])

        if spectral_oe_train or spectral_oe_cal:
            spectral_copy.transform = generic_norm_transform(mean_partial, std_partial)
            train_spec = spectral_copy
            if loss_name == "fcdd":
                cal_spec = copy.deepcopy(train_spec)
                cal_spec.switch_target_type()
            else:
                cal_spec = train_spec
        if not spectral_oe_train or not spectral_oe_cal:
            oe_copy.dataset.transform = train_transform_partial

        train_loader_partial, cal_loader = None, None
        if base_loss_unsupervised:
            train_loader_partial = DataLoader(train_data_partial, 128, num_workers=4)
            if spectral_oe_cal:
                cal_loader = BalancedLoader(cal_data_indist, cal_spec)
            else:
                oe_copy.dataset.target_transform = synthetic_mask(
                    imshape[-2:], anomaly_targets=True
                )
                cal_loader = BalancedLoader(cal_data_indist, oe_copy)
        else:
            if spectral_oe_train and spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, train_spec)
                cal_loader = BalancedLoader(cal_data_indist, cal_spec)
            elif spectral_oe_train and not spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, train_spec)
                oe_copy.dataset.target_transform = synthetic_mask(
                    imshape[-2:], anomaly_targets=True
                )
                cal_loader = BalancedLoader(cal_data_indist, oe_copy)
            elif not spectral_oe_train and spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, oe_copy)
                cal_loader = BalancedLoader(cal_data_indist, cal_spec)
            else:
                oe_train, oe_cal = get_train_cal_splits(
                    oe_copy, train_cal_splits_dir / f"{basename}-oe.json"
                )
                if loss_name == "fcdd":
                    oe_cal = copy.deepcopy(oe_cal)
                    oe_cal.dataset.target_transform = synthetic_mask(
                        imshape[-2:], anomaly_targets=True
                    )
                train_loader_partial = BalancedLoader(train_data_partial, oe_train)
                cal_loader = BalancedLoader(cal_data_indist, oe_cal)

        # Prepare test data with partial mean, std norm
        test_copy.dataset.transform = test_transform_partial
        test_loader_ph = DataLoader(test_copy, 128, num_workers=4)

        # ---------

        # Train partial
        partial_pre = f"{basename}-partial-{oe_train_type}"

        model_partial = SEG_MODEL_MAP[loss_name]
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model_partial.parameters())
        sched = sched(opt)

        model_partial = train(
            epochs=epochs,
            net=model_partial,
            loss_function=base_loss,
            opt=opt,
            sched=sched,
            dloader=train_loader_partial,
            savename=partial_pre,
            savedir=model_dir,
        )

        # -------

        logits_fn, pests_fn = base_loss.get_logits, base_loss.get_pests
        inputs_to_logits_fn, inputs_to_pests_fn = (
            lambda inputs: logits_fn(
                model_inputs=inputs, model_outputs=model_partial(inputs)
            ),
            lambda inputs: pests_fn(
                model_inputs=inputs, model_outputs=model_partial(inputs)
            ),
        )
        modules_phcal = [model_partial, base_loss]

        # Calibrate - Platt
        platt_pre = f"{partial_pre}-platt-{oe_cal_type}"
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        pm = apply_posthoc_calibration_seg(
            "platt",
            inputs_to_logits_fn,
            cal_loader,
            epochs,
            opt,
            sched,
            wh_shape=imshape[-2:],
            modules_in_fn=modules_phcal,
            savepath=model_dir / f"{platt_pre}.pt",
        )
        # Test - normal
        results_path = results_dir / f"{platt_pre}.json"
        inputs_to_anomaly_score_platt = lambda inputs: pm(inputs_to_logits_fn(inputs))
        modules_platt = modules_phcal + [pm]
        evaluate_thresholding_localization(
            inputs_to_anomaly_score_platt,
            test_loader_ph,
            modules_platt,
            results_path,
        )

        # Test - with input perturbation
        results_path = results_dir / f"{platt_pre}-perturb.json"
        inputs_to_loss_platt = (
            lambda inputs, labels: F.binary_cross_entropy_with_logits(
                pm(inputs_to_logits_fn(inputs)), labels
            )
        )
        evaluate_thresholding_localization_perturbation(
            inputs_to_anomaly_score_platt,
            inputs_to_loss_platt,
            std_partial,
            test_loader_ph,
            modules=modules_platt,
            savepath=results_path,
        )

        # Calibrate - Beta
        beta_pre = f"{partial_pre}-beta-{oe_cal_type}"
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        bm = apply_posthoc_calibration_seg(
            "beta",
            inputs_to_pests_fn,
            cal_loader,
            epochs,
            opt,
            sched,
            wh_shape=imshape[-2:],
            modules_in_fn=modules_phcal,
            savepath=model_dir / f"{beta_pre}.pt",
        )
        # Test - normal
        results_path = results_dir / f"{beta_pre}.json"
        inputs_to_anomaly_score_beta = lambda inputs: bm(
            inputs_to_pests_fn(inputs)
        )  # NB: bm output are calibrated logits
        modules_beta = modules_phcal + [bm]
        evaluate_thresholding_localization(
            inputs_to_anomaly_score_beta,
            test_loader_ph,
            modules=modules_beta,
            savepath=results_path,
        )

        # Test - with input perturbation
        results_path = results_dir / f"{beta_pre}-perturb.json"
        inputs_to_loss_beta = lambda inputs, labels: F.binary_cross_entropy_with_logits(
            bm(inputs_to_pests_fn(inputs)), labels
        )
        evaluate_thresholding_localization_perturbation(
            inputs_to_anomaly_score_beta,
            inputs_to_loss_beta,
            std_partial,
            test_loader_ph,
            modules=modules_beta,
            savepath=results_path,
        )

        # ---------

        # Train full
        full_pre = f"{basename}-full-{oe_train_type}"

        model_full = SEG_MODEL_MAP[loss_name]
        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model_full.parameters())
        sched = sched(opt)

        model_full = train(
            epochs=epochs,
            net=model_full,
            loss_function=base_loss,
            opt=opt,
            sched=sched,
            dloader=train_loader_full,
            savename=full_pre,
            savedir=model_dir,
        )
        # Test full - normal
        inputs_to_anomaly_score_full = lambda inputs: anomaly_score(
            model_inputs=inputs, model_outputs=model_full(inputs)
        )
        modules_full = [model_full, anomaly_score]

        results_path = results_dir / f"{full_pre}.json"
        evaluate_thresholding_localization(
            inputs_to_anomaly_score_full,
            test_loader,
            modules=modules_full,
            savepath=results_path,
        )
        # Test full - with input perturbation
        results_path = results_dir / f"{full_pre}-perturb.json"
        inputs_to_loss_full = lambda inputs, labels: base_loss(
            model_inputs=inputs, model_outputs=model_full(inputs), labels=labels
        )
        modules_pert_full = modules_full + [base_loss]
        evaluate_thresholding_localization_perturbation(
            inputs_to_anomaly_score_full,
            inputs_to_loss_full,
            std_full,
            test_loader,
            detection_targets_for_loss=loss_name == "fcdd",
            modules=modules_pert_full,
            savepath=results_path,
        )
