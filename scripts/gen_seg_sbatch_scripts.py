import __init__  # noqa
from pathlib import Path
import argparse

from phcad.train.losses import SEG_LOSS_MAP
from phcad.data.constants import DS_TO_LABELS_MAP
from constants import SLURMDIR

slurm_subdir = SLURMDIR / "localization"
parser = argparse.ArgumentParser()
parser.add_argument("dataset_name")
parser.add_argument("-t", "--test-one", action="store_true")


def generate_batch(dataset_name, test_one_label):
    sbpaths = []
    for loss_name in SEG_LOSS_MAP.keys():
        sbpath = generate_sbatch(dataset_name, loss_name, test_one_label=test_one_label)
        sbpaths.append(sbpath)
        sbpath = generate_sbatch(
            dataset_name, loss_name, spectral_oe_cal=True, test_one_label=test_one_label
        )
        sbpaths.append(sbpath)
    postfix = "test-one" if test_one_label else "all"
    with open(slurm_subdir / f"{dataset_name}-{postfix}.sh", "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write("sbatch " + "\nsbatch ".join(map(str, sbpaths)))
        f.write("\n")


def generate_sbatch(
    dataset_name,
    loss_name,
    spectral_oe_train=False,
    spectral_oe_cal=False,
    test_one_label=False,
):
    prefix = """#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH -t 1-0  # 1 day
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpuidle
"""
    infix = """module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate ml
"""
    postfix = "conda deactivate\n"

    sbatch_dir = slurm_subdir / dataset_name
    logdir = sbatch_dir / "logs"
    errdir = sbatch_dir / "error-logs"
    for dir in [logdir, errdir]:
        if not dir.exists():
            dir.mkdir(parents=True)

    oe_train = "spec" if spectral_oe_train else "oe"
    oe_cal = "spec" if spectral_oe_cal else "oe"

    log_name = f"{loss_name}-{oe_train}-{oe_cal}"
    if test_one_label:
        log_path = logdir / f"{log_name}_%j.log"
        err_path = errdir / f"{log_name}_%j.log"
    else:
        log_path = logdir / f"{log_name}_%a_%A.log"
        err_path = errdir / f"{log_name}_%a_%A_err.log"

    job_name = f"seg-{dataset_name}-{log_name}"
    dynamic_sbatch_args = (
        f"#SBATCH -J {job_name}\n"
        f"#SBATCH --output={log_path}\n"
        f"#SBATCH --error={err_path}\n"
    )
    if not test_one_label:
        dynamic_sbatch_args += (
            f"#SBATCH --array=0-{len(DS_TO_LABELS_MAP[dataset_name]) - 1}\n"
        )
    dynamic_sbatch_args += "\n"

    seg_script_path = Path(f"{__file__}/../run_localization.py").resolve()
    cmd_args = f"{dataset_name} {loss_name}"
    if test_one_label:
        cmd_args += " 0"
    else:
        cmd_args += " $SLURM_ARRAY_TASK_ID"
    if spectral_oe_train:
        cmd_args += " --spectral-train"
    if spectral_oe_cal:
        cmd_args += " --spectral-cal"
    dynamic_cmd_line = f"python {seg_script_path} {cmd_args}\n"

    sbatch_script = prefix + dynamic_sbatch_args + infix + dynamic_cmd_line + postfix
    postfix = "test-one" if test_one_label else "all"
    sbatch_savepath = sbatch_dir / f"{log_name}-{postfix}.sbatch"
    with open(sbatch_savepath, "w") as f:
        f.write(sbatch_script)
    return sbatch_savepath


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset_name not in DS_TO_LABELS_MAP.keys():
        raise ValueError(f"dataset_name must be one of {list(DS_TO_LABELS_MAP.keys())}")

    generate_batch(args.dataset_name, args.test_one)
