import json
import numpy as np
from collections import defaultdict

from phcad.data.constants import DS_TO_LABELS_MAP
from phcad.experiments.constants import EXPROOT, NUMSEEDS


num_interpolation_points = 101
cal_methods = ["beta", "platt"]
det_cal_methods = cal_methods + ["phtrain"]
cal_data = ["oe", "spectral"]


def parse_results(dataset_name, loss_name, experiment_type="detection"):
    results_dir = EXPROOT / experiment_type / dataset_name / loss_name / "results"
    labels = DS_TO_LABELS_MAP[dataset_name]
    base_loss_unsupervised = loss_name == "dsvdd" or loss_name == "ssim"
    cal_train = "none" if base_loss_unsupervised else "oe"

    auroc_map = {}
    auroc_map["all"] = defaultdict(list)
    auroc_map["avg"] = {}
    auroc_map["seeds"] = {}
    auroc_map["roc"] = {}
    aupro_map = {}
    aupro_map["all"] = defaultdict(list)
    aupro_map["avg"] = {}
    aupro_map["seeds"] = {}
    aupro_map["pro"] = {}
    for label in labels:
        auroc_map["avg"][label] = {}
        auroc_map["seeds"][label] = {}
        aupro_map["avg"][label] = {}
        aupro_map["seeds"][label] = {}
        # num_test_samples = len(get_dataset(dataset_name, "test", label))
        for datamode in ["full", "partial"]:
            infixes = []
            if datamode == "full":
                base = f"{datamode}-{cal_train}"
                infixes += [base, f"{base}-perturb"]
            elif datamode == "partial":
                cal_list = (
                    det_cal_methods if experiment_type == "detection" else cal_methods
                )
                for cal_method in cal_list:
                    for cal_d in cal_data:
                        base = f"{datamode}-{cal_train}-{cal_method}-{cal_d}"
                        infixes += [base, f"{base}-perturb"]

            for infix in infixes:
                aurocs = np.empty(NUMSEEDS)
                rocs = np.zeros((NUMSEEDS, 2, num_interpolation_points))
                if experiment_type == "localization":
                    aupros = np.empty(NUMSEEDS)
                    pros = np.zeros((NUMSEEDS, 2, num_interpolation_points))

                for seed in range(NUMSEEDS):
                    fname = f"{label}-{seed}-{infix}.json"
                    with open(results_dir / fname, "rb") as f:
                        results = json.loads(f.read())
                    auroc, roc = (
                        results["auroc"],
                        [results["roc"][ax] for ax in ["x", "y"]],
                    )

                    aurocs[seed] = auroc
                    if experiment_type == "detection":
                        fpr = np.linspace(0, 1, num_interpolation_points)
                        tpr = np.interp(fpr, roc[0], roc[1])
                        rocs[seed, 0, :] += fpr
                        rocs[seed, 1, :] += tpr
                    else:
                        fpr, tpr = roc[0], roc[1]
                        aupro, pro = (
                            results["aupro"],
                            np.array([results["pro"][ax] for ax in ["x", "y"]]),
                        )
                        aupros[seed] = aupro
                        pros[seed, 0, :] += pro[0, :]
                        pros[seed, 1, :] += pro[1, :]

                auroc_map["all"][infix] += [aurocs.mean()]
                auroc_map["avg"][label][f"{label}-{infix}"] = aurocs.mean()
                auroc_map["seeds"][label][f"{label}-{infix}"] = aurocs.tolist()

                if infix not in auroc_map["roc"]:
                    auroc_map["roc"][infix] = defaultdict(list)
                auroc_map["roc"][infix]["x"] += [rocs[:, 0, :].mean(0)]
                auroc_map["roc"][infix]["y"] += [rocs[:, 1, :].mean(0)]

                if experiment_type == "localization":
                    aupro_map["all"][infix] += [aupros.mean()]
                    aupro_map["avg"][label][f"{label}-{infix}"] = aupros.mean()
                    aupro_map["seeds"][label][f"{label}-{infix}"] = aupros.tolist()

                    if infix not in aupro_map["pro"]:
                        aupro_map["pro"][infix] = defaultdict(list)
                    aupro_map["pro"][infix]["x"] += [pros[:, 0, :].mean(0)]
                    aupro_map["pro"][infix]["y"] += [pros[:, 1, :].mean(0)]

    all_map = auroc_map["all"]
    for infix in all_map.keys():
        all_map[infix] = np.array(all_map[infix]).mean().tolist()
        auroc_map["roc"][infix]["x"] = (
            np.stack(auroc_map["roc"][infix]["x"]).mean(0).tolist()
        )
        auroc_map["roc"][infix]["y"] = (
            np.stack(auroc_map["roc"][infix]["y"]).mean(0).tolist()
        )

        if experiment_type == "localization":
            aupro_map["all"][infix] = np.array(aupro_map["all"][infix]).mean().tolist()
            aupro_map["pro"][infix]["x"] = (
                np.stack(aupro_map["pro"][infix]["x"]).mean(0).tolist()
            )
            aupro_map["pro"][infix]["y"] = (
                np.stack(aupro_map["pro"][infix]["y"]).mean(0).tolist()
            )
    with open(EXPROOT / experiment_type / dataset_name / f"{loss_name}.json", "w") as f:
        f.write(json.dumps(auroc_map, indent=2))
    if experiment_type == "localization":
        with open(
            EXPROOT / experiment_type / dataset_name / f"{loss_name}-pro.json", "w"
        ) as f:
            f.write(json.dumps(aupro_map, indent=2))


def parse_cal_curves(dataset_name, loss_name, experiment_type="detection"):
    results_dir = EXPROOT / experiment_type / dataset_name / loss_name / "cal_curves"
    labels = DS_TO_LABELS_MAP[dataset_name]
    base_loss_unsupervised = loss_name == "dsvdd" or loss_name == "ssim"
    cal_train = "none" if base_loss_unsupervised else "oe"
    numbins = 10

    curve_map = {}
    curve_map["all"] = {}
    curve_map["all"]["ece"] = defaultdict(list)
    curve_map["all"]["mce"] = defaultdict(list)
    curve_map["all"]["curve"] = {}
    curve_map["avg"] = {}
    curve_map["seeds"] = {}

    zvals_master = defaultdict(list)
    for label in labels:
        curve_map["avg"][label] = {}
        curve_map["avg"][label]["ece"] = {}
        curve_map["avg"][label]["mce"] = {}
        curve_map["seeds"][label] = {}
        curve_map["seeds"][label]["ece"] = {}
        curve_map["seeds"][label]["mce"] = {}
        for datamode in ["full", "partial"]:
            infixes = []
            if datamode == "full":
                base = f"{datamode}-{cal_train}"
                infixes += [base]
            elif datamode == "partial":
                cal_list = (
                    det_cal_methods if experiment_type == "detection" else cal_methods
                )
                for cal_method in cal_list:
                    for cal_d in cal_data:
                        base = f"{datamode}-{cal_train}-{cal_method}-{cal_d}"
                        infixes += [base]

            for infix in infixes:
                if infix not in curve_map["all"]["curve"]:
                    curve_map["all"]["curve"][infix] = defaultdict(list)

                confs = np.zeros((NUMSEEDS, numbins))
                accs = np.zeros((NUMSEEDS, numbins))
                eces = np.empty(NUMSEEDS)
                mces = np.empty(NUMSEEDS)
                zidcs = np.empty((NUMSEEDS, numbins))
                for seed in range(NUMSEEDS):
                    fname = f"{label}-{seed}-{infix}.json"
                    with open(results_dir / fname, "rb") as f:
                        results = json.loads(f.read())

                    for k, v in results.items():
                        if "total" in k:
                            tbc = np.array(v)
                        if "indist" in k:
                            ibc = np.array(v)
                        if "summed" in k:
                            sbp = np.array(v)

                    print(tbc)
                    zidx = tbc < 10
                    zidcs[seed] = zidx

                    conf = sbp[~zidx] / tbc[~zidx]
                    acc = ibc[~zidx] / tbc[~zidx]
                    cal_errors = np.abs(conf - acc)
                    ece = (tbc[~zidx] / tbc.sum() * cal_errors).mean()
                    mce = cal_errors.max()
                    eces[seed] = ece
                    mces[seed] = mce

                    accs[seed, ~zidx] = acc
                    confs[seed, ~zidx] = conf

                curve_map["all"]["ece"][infix] += [eces.mean()]
                curve_map["all"]["mce"][infix] += [mces.mean()]
                curve_map["avg"][label]["ece"][f"{label}-{infix}"] = float(eces.mean())
                curve_map["seeds"][label]["ece"][f"{label}-{infix}"] = eces.tolist()
                curve_map["avg"][label]["mce"][f"{label}-{infix}"] = float(mces.mean())
                curve_map["seeds"][label]["mce"][f"{label}-{infix}"] = mces.tolist()

                num_nonzero_vals_in_bins = NUMSEEDS - zidcs.sum(0)
                zdiv_idcs = num_nonzero_vals_in_bins == 0
                zvals_master[infix] += [zdiv_idcs]

                mean_accs = np.zeros(numbins)
                mean_accs[~zdiv_idcs] = (
                    accs[:, ~zdiv_idcs].sum(0) / num_nonzero_vals_in_bins[~zdiv_idcs]
                )
                mean_confs = np.zeros(numbins)
                mean_confs[~zdiv_idcs] = (
                    confs[:, ~zdiv_idcs].sum(0) / num_nonzero_vals_in_bins[~zdiv_idcs]
                )

                curve_map["all"]["curve"][infix]["conf"] += [mean_confs]
                curve_map["all"]["curve"][infix]["acc"] += [mean_accs]

    all_map = curve_map["all"]
    for infix in all_map["ece"].keys():
        all_map["ece"][infix] = np.array(all_map["ece"][infix]).mean().tolist()
        all_map["mce"][infix] = np.array(all_map["mce"][infix]).mean().tolist()
        inf_curve_map = all_map["curve"][infix]
        for key in inf_curve_map:
            curve_vals = np.stack(inf_curve_map[key])
            zidcs_per_label = np.stack(zvals_master[infix])
            num_nonzero_vals_in_bins = len(labels) - zidcs_per_label.sum(0)
            zidcs = num_nonzero_vals_in_bins == 0

            mean_curve_vals = np.zeros(numbins)  # over labels for given infix
            mean_curve_vals[~zidcs] = (
                curve_vals[:, ~zidcs].sum(0) / num_nonzero_vals_in_bins[~zidcs]
            )

            inf_curve_map[key] = mean_curve_vals.tolist()
    with open(
        EXPROOT / experiment_type / dataset_name / f"{loss_name}-cal.json", "w"
    ) as f:
        f.write(json.dumps(curve_map, indent=2))
