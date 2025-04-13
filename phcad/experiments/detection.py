import copy

from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F

from phcad.data.constants import OE_DATASET_MAP, MVTEC_LABELS_NOFLIP
from phcad.data.spectral_natural_images import SpectralNaturalImages
from phcad.data.transforms import (
    TRAIN_TRANSFORM_MAP,
    TEST_TRANSFORM_MAP,
    generic_norm_transform,
    label_to_zero,
    label_to_one,
    mask_to_class,
)
from phcad.data.utils import (
    get_dataset,
    get_train_cal_splits,
    mean_std,
    BalancedLoader,
    DATASET_MAP,
)
from phcad.experiments.constants import EXPROOT, NUMSEEDS
from phcad.models.constants import MODEL_MAP
from phcad.train.losses import LOSS_MAP
from phcad.train.train import train
from phcad.train.calibrate import apply_posthoc_calibration
from phcad.test.anomaly_scores import ANOMALY_SCORES
from phcad.test.evaluate import (
    evaluate_thresholding,
    evaluate_thresholding_perturbation,
)
from phcad.train.utils import get_optim_sched_epochs
from phcad.utils import dsvdd_center


def run_detection_experiment(
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

    anomaly_score = ANOMALY_SCORES[loss_name]
    if loss_name == "ssim":
        if dataset_name == "cifar10" or dataset_name == "fmnist":
            anomaly_score = anomaly_score(win_size=11)
        else:
            anomaly_score = anomaly_score(win_size=11)

    # Setup save directories
    exp_dir = EXPROOT / "detection" / dataset_name / loss_name
    model_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
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
    flip = True
    if "mvtec" in dataset_name and label in MVTEC_LABELS_NOFLIP:
        flip = False
    train_transform_full = TRAIN_TRANSFORM_MAP[transform_and_model_identifier](
        mean_full, std_full, ae=loss_name == "ssim", flip=flip
    )
    test_transform_full = TEST_TRANSFORM_MAP[transform_and_model_identifier](
        mean_full, std_full, ae=loss_name == "ssim"
    )

    train_full.dataset.transform = train_transform_full
    train_full.dataset.target_transform = label_to_zero

    oe_data_full = None
    if not (spectral_oe_train and spectral_oe_cal):
        oe_name = OE_DATASET_MAP[dataset_name]
        if oe_name == "cifar100":
            oe_train_1 = get_dataset(OE_DATASET_MAP[dataset_name], "train")
            oe_train_2 = get_dataset(OE_DATASET_MAP[dataset_name], "test")
            for oe_dataset in [oe_train_1, oe_train_2]:
                oe_dataset.dataset.transform = train_transform_full
                oe_dataset.dataset.target_transform = label_to_one
            oe_data_full = oe_train_1 + oe_train_2
        elif oe_name == "imagenet21k-minus1k":
            oe_data_full = get_dataset(oe_name, "train")
            oe_data_full.dataset.transform = train_transform_full
            oe_data_full.dataset.target_transform = label_to_one

    spectral_data = None
    if spectral_oe_train or spectral_oe_cal:
        imshape = train_full[0][0].shape
        spectral_data = SpectralNaturalImages(
            imshape,
            transform=generic_norm_transform(mean_full, std_full),
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

    if dataset_type == "classification":
        # detection
        test_data_in = get_dataset(dataset_name, "test", label)
        test_data_in.dataset.transform = test_transform_full
        test_data_in.dataset.target_transform = label_to_zero
        test_data_anom = get_dataset(dataset_name, "test", label, complement=True)
        test_data_anom.dataset.transform = test_transform_full
        test_data_anom.dataset.target_transform = label_to_one
        test_data = test_data_in + test_data_anom
        test_loader = DataLoader(test_data, 128, num_workers=4)
    elif dataset_type == "localization":
        # localized, subtle anomalies
        test_data = get_dataset(dataset_name, "test", label)
        test_data.dataset.transform = test_transform_full
        test_data.dataset.target_transform = mask_to_class
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
        train_transform_partial = TRAIN_TRANSFORM_MAP[transform_and_model_identifier](
            mean_partial, std_partial, ae=loss_name == "ssim", flip=flip
        )
        test_transform_partial = TEST_TRANSFORM_MAP[transform_and_model_identifier](
            mean_partial, std_partial, ae=loss_name == "ssim"
        )

        for ds in [train_data_partial, cal_data_indist]:
            ds.dataset.transform = train_transform_partial

        if spectral_oe_train or spectral_oe_cal:
            spectral_copy.transform = generic_norm_transform(mean_partial, std_partial)
        if not spectral_oe_train or not spectral_oe_cal:
            if isinstance(oe_copy, ConcatDataset):
                for subset in oe_copy.datasets:
                    subset.dataset.transform = train_transform_partial
            elif isinstance(oe_copy, Subset):
                oe_copy.dataset.transform = train_transform_partial

        train_loader_partial, cal_loader = None, None
        if base_loss_unsupervised:
            train_loader_partial = DataLoader(train_data_partial, 128, num_workers=4)
            if spectral_oe_cal:
                cal_loader = BalancedLoader(cal_data_indist, spectral_copy)
            else:
                cal_loader = BalancedLoader(cal_data_indist, oe_copy)
        else:
            if spectral_oe_train and spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, spectral_copy)
                cal_loader = BalancedLoader(cal_data_indist, spectral_copy)
            elif spectral_oe_train and not spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, spectral_copy)
                cal_loader = BalancedLoader(cal_data_indist, oe_copy)
            elif not spectral_oe_train and spectral_oe_cal:
                train_loader_partial = BalancedLoader(train_data_partial, oe_copy)
                cal_loader = BalancedLoader(cal_data_indist, spectral_copy)
            else:
                oe_train, oe_cal = get_train_cal_splits(
                    oe_copy, train_cal_splits_dir / f"{basename}-oe.json"
                )
                train_loader_partial = BalancedLoader(train_data_partial, oe_train)
                cal_loader = BalancedLoader(cal_data_indist, oe_cal)

        # Prepare test data with partial mean, std norm
        if isinstance(test_copy, ConcatDataset):
            for subset in test_copy.datasets:
                subset.dataset.transform = test_transform_partial
        elif isinstance(test_copy, Subset):
            test_copy.dataset.transform = test_transform_partial

        test_loader_ph = DataLoader(test_copy, 128, num_workers=4)

        # ---------

        # Train partial
        partial_pre = f"{basename}-partial-{oe_train_type}"

        model_partial = MODEL_MAP[transform_and_model_identifier](**base_model_args)
        if loss_name == "dsvdd":
            center_savepath = model_dir / f"{partial_pre}-center.pt"
            center = dsvdd_center(model_partial, train_loader_partial, center_savepath)
            loss_partial = base_loss(center=center)
        else:
            loss_partial = base_loss

        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model_partial.parameters())
        sched = sched(opt)

        model_partial = train(
            epochs=epochs,
            net=model_partial,
            loss_function=loss_partial,
            opt=opt,
            sched=sched,
            dloader=train_loader_partial,
            savename=partial_pre,
            savedir=model_dir,
        )

        # ---------

        # Post-hoc training
        phtrain_pre = f"{partial_pre}-phtrain-{oe_cal_type}"
        model_phtrain = copy.deepcopy(model_partial)
        model_phtrain.prepare_calibration_network()
        loss_phtrain = LOSS_MAP["bce"]

        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        try:
            opt = opt(model_phtrain.layers.calibration_head.parameters())
        except AttributeError:
            opt = opt(model_phtrain.calibration_head.parameters())  # WRN18 hack
        sched = sched(opt)

        model_phtrain = train(
            epochs=epochs,
            net=model_phtrain,
            loss_function=loss_phtrain,
            opt=opt,
            sched=sched,
            dloader=cal_loader,
            savename=phtrain_pre,
            savedir=model_dir,
        )
        # Test - normal
        results_path = results_dir / f"{phtrain_pre}.json"
        inputs_to_anomaly_score_phtrain = lambda inputs: F.sigmoid(
            model_phtrain(inputs)
        )
        evaluate_thresholding(
            inputs_to_anomaly_score_phtrain,
            test_loader_ph,
            [model_phtrain],
            results_path,
        )
        # Test - with input perturbation
        results_path = results_dir / f"{phtrain_pre}-perturb.json"
        inputs_to_loss_phtrain = (
            lambda inputs, labels: F.binary_cross_entropy_with_logits(
                model_phtrain(inputs), labels
            )
        )
        evaluate_thresholding_perturbation(
            inputs_to_anomaly_score_phtrain,
            inputs_to_loss_phtrain,
            std_partial,
            test_loader_ph,
            [model_phtrain],
            results_path,
        )

        # -------

        logits_fn, pests_fn = loss_partial.get_logits, loss_partial.get_pests
        inputs_to_logits_fn, inputs_to_pests_fn = (
            lambda inputs: logits_fn(
                model_inputs=inputs, model_outputs=model_partial(inputs)
            ),
            lambda inputs: pests_fn(
                model_inputs=inputs, model_outputs=model_partial(inputs)
            ),
        )
        modules_phcal = [model_partial, loss_partial]

        # Calibrate - Platt
        platt_pre = f"{partial_pre}-platt-{oe_cal_type}"
        pm = apply_posthoc_calibration(
            "platt",
            inputs_to_logits_fn,
            cal_loader,
            model_dir / f"{platt_pre}.pt",
            modules_in_fn=modules_phcal,
        )
        # Test - normal
        results_path = results_dir / f"{platt_pre}.json"
        inputs_to_anomaly_score_platt = lambda inputs: pm(inputs_to_logits_fn(inputs))
        modules_platt = modules_phcal + [pm]
        evaluate_thresholding(
            inputs_to_anomaly_score_platt, test_loader_ph, modules_platt, results_path
        )

        # Test - with input perturbation
        results_path = results_dir / f"{platt_pre}-perturb.json"
        inputs_to_loss_platt = (
            lambda inputs, labels: F.binary_cross_entropy_with_logits(
                pm(inputs_to_logits_fn(inputs)), labels
            )
        )
        evaluate_thresholding_perturbation(
            inputs_to_anomaly_score_platt,
            inputs_to_loss_platt,
            std_partial,
            test_loader_ph,
            modules_platt,
            results_path,
        )

        # Calibrate - Beta
        beta_pre = f"{partial_pre}-beta-{oe_cal_type}"
        bm = apply_posthoc_calibration(
            "beta",
            inputs_to_pests_fn,
            cal_loader,
            model_dir / f"{beta_pre}.pt",
            modules_in_fn=modules_phcal,
        )
        # Test - normal
        results_path = results_dir / f"{beta_pre}.json"
        inputs_to_anomaly_score_beta = lambda inputs: bm(
            inputs_to_pests_fn(inputs)
        )  # Note bm outputs logits
        modules_beta = modules_phcal + [bm]
        evaluate_thresholding(
            inputs_to_anomaly_score_beta, test_loader_ph, modules_beta, results_path
        )

        # Test - with input perturbation
        results_path = results_dir / f"{beta_pre}-perturb.json"
        inputs_to_loss_beta = lambda inputs, labels: F.binary_cross_entropy_with_logits(
            bm(inputs_to_pests_fn(inputs)), labels
        )
        evaluate_thresholding_perturbation(
            inputs_to_anomaly_score_beta,
            inputs_to_loss_beta,
            std_partial,
            test_loader_ph,
            modules_beta,
            results_path,
        )

        # ---------

        # Train full
        full_pre = f"{basename}-full-{oe_train_type}"

        model_full = MODEL_MAP[transform_and_model_identifier](**base_model_args)
        if loss_name == "dsvdd":
            center_savepath = model_dir / f"{full_pre}-center.pt"
            center = dsvdd_center(model_full, train_loader_full, center_savepath)
            loss_full = base_loss(center=center)
            anomaly_score_full = anomaly_score(center=center)
        else:
            loss_full = base_loss
            anomaly_score_full = anomaly_score

        opt, sched, epochs = get_optim_sched_epochs(dataset_name)
        opt = opt(model_full.parameters())
        sched = sched(opt)

        model_full = train(
            epochs=epochs,
            net=model_full,
            loss_function=loss_full,
            opt=opt,
            sched=sched,
            dloader=train_loader_full,
            savename=full_pre,
            savedir=model_dir,
        )
        # Test full - normal
        inputs_to_anomaly_score_full = lambda inputs: anomaly_score_full(
            model_inputs=inputs, model_outputs=model_full(inputs)
        )
        modules_full = [model_full, anomaly_score_full]

        results_path = results_dir / f"{full_pre}.json"
        evaluate_thresholding(
            inputs_to_anomaly_score_full, test_loader, modules_full, results_path
        )
        # Test full - with input perturbation
        results_path = results_dir / f"{full_pre}-perturb.json"
        inputs_to_loss_full = lambda inputs, labels: loss_full(
            model_inputs=inputs, model_outputs=model_full(inputs), labels=labels
        )
        modules_pert_full = modules_full + [loss_full]
        evaluate_thresholding_perturbation(
            inputs_to_anomaly_score_full,
            inputs_to_loss_full,
            std_full,
            test_loader,
            modules_pert_full,
            results_path,
        )
