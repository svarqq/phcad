import random
import json
import logging
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import v2

from phcad.data.constants import (
    DATADIR,
    FMNIST_LABEL_MAP,
    CIFAR10_LABEL_MAP,
    MVTEC_LABEL_MAP,
    MPDD_LABEL_MAP,
)
from phcad.data.imagenet import ImageNet21KMinus1K
from phcad.data.mvtec_mpdd import MVTecMPDD


MVTec = partial(MVTecMPDD, "mvtec")
MPDD = partial(MVTecMPDD, "mpdd")


DATASET_MAP = {
    "fmnist": (FashionMNIST, FMNIST_LABEL_MAP, "classification"),
    "cifar10": (CIFAR10, CIFAR10_LABEL_MAP, "classification"),
    "cifar100": (CIFAR100, None, "classification"),
    "imagenet21k-minus1k": (ImageNet21KMinus1K, None, "classification"),
    "mvtec": (MVTec, MVTEC_LABEL_MAP, "localization"),
    "mpdd": (MPDD, MPDD_LABEL_MAP, "localization"),
}

logger = logging.getLogger(__name__)


def get_train_cal_splits(dataset, idcs_savepath=None, datadir=DATADIR):
    train_data, cal_data = None, None

    if idcs_savepath and idcs_savepath.exists():
        logger.info(f"Loading train-cal split indices from {idcs_savepath}")
        with open(idcs_savepath, "r") as f:
            idcs = json.loads(f.read())

        train_data, cal_data = None, None
        if isinstance(dataset, Subset):
            train_idcs, cal_idcs = idcs["train"], idcs["cal"]
            train_data = Subset(dataset.dataset, train_idcs)
            cal_data = Subset(dataset.dataset, cal_idcs)
        elif isinstance(dataset, ConcatDataset):
            train, cal = [], []
            for i, ds in enumerate(dataset.datasets):
                train_idcs, cal_idcs = idcs[f"train-{i}"], idcs[f"cal-{i}"]
                train_subs = Subset(ds.dataset, train_idcs)
                cal_subs = Subset(ds.dataset, cal_idcs)
                train.append(train_subs)
                cal.append(cal_subs)
            train_data = ConcatDataset(train)
            cal_data = ConcatDataset(cal)

    else:
        # 3:1 split
        split_idcs = {}
        if isinstance(dataset, Subset):
            num_train = (len(dataset) * 3) // 4
            indices = dataset.indices
            train_idcs = random.sample(indices, num_train)
            cal_idcs = list(set(indices) - set(train_idcs))
            train_data = Subset(dataset.dataset, train_idcs)
            cal_data = Subset(dataset.dataset, cal_idcs)
            split_idcs = {"train": train_idcs, "cal": cal_idcs}
        elif isinstance(dataset, ConcatDataset):
            train, cal = [], []
            for i, ds in enumerate(dataset.datasets):
                num_train = (len(ds) * 3) // 4
                indices = ds.indices
                train_idcs = random.sample(indices, num_train)
                cal_idcs = list(set(indices) - set(train_idcs))
                train_subs = Subset(ds.dataset, train_idcs)
                cal_subs = Subset(ds.dataset, cal_idcs)

                train.append(train_subs)
                cal.append(cal_subs)
                split_idcs[f"train-{i}"] = train_idcs
                split_idcs[f"cal-{i}"] = cal_idcs
            train_data = ConcatDataset(train)
            cal_data = ConcatDataset(cal)

        if idcs_savepath and not idcs_savepath.exists():
            with open(idcs_savepath, "w") as f:
                f.write(json.dumps(split_idcs))

    return train_data, cal_data


def get_dataset(
    dataset_name: str,
    split: str,
    label: str | None = None,
    complement=False,
    test_indist_only=False,
    datadir=DATADIR,
) -> Subset:
    # Returned Subset always has VisionDataset as an object attribute here
    if dataset_name not in DATASET_MAP.keys():
        raise ValueError(
            f"Parameter dataset_name must be one of {list(DATASET_MAP.keys())}"
        )
    if split != "train" and split != "test":
        raise ValueError('Parameter split must be one of ["train", "test"]')
    if complement and split == "train":
        raise ValueError(
            "Cannot set parameter complement to True for train data, it is designed for detection testing"
        )

    dataset_class, label_map, dataset_type = DATASET_MAP[dataset_name]
    if label and not label_map:
        raise ValueError(
            f"{dataset_name} is not valid for one-class data, label must be None"
        )
    elif label and label not in label_map.keys():
        raise ValueError(
            f"Label {label} is not in dataset {dataset_name}. Choose one of {list(label_map.keys())}"
        )
    if split == "test" and dataset_type == "localization" and complement:
        error_msg = (
            "Defensively disallowing complement of test localization data from "
            "being fetched, as it's designed for detection testing"
        )
        raise ValueError(error_msg)

    if dataset_name == "imagenet21k-minus1k":
        parentdir = "imagenet21k"
        dataset_dir = datadir / parentdir
        oe_data = ImageNet21KMinus1K(root=dataset_dir)
        return Subset(oe_data, list(range(len(oe_data))))
    else:
        parentdir = dataset_name
    dataset_dir = datadir / parentdir

    args = {"root": dataset_dir}
    args["train"] = True if split == "train" else False
    args["download"] = True

    if dataset_type == "classification":
        full_data = dataset_class(**args)
        if not label:
            return Subset(full_data, list(range(len(full_data))))

        labels = np.asarray(full_data.targets)
        if not complement:
            label_idcs = (
                np.argwhere(labels == full_data.class_to_idx[label_map[label]])
                .flatten()
                .tolist()
            )
        else:
            label_idcs = (
                np.argwhere(labels != full_data.class_to_idx[label_map[label]])
                .flatten()
                .tolist()
            )
        label_data = Subset(full_data, label_idcs)
        return label_data

    elif dataset_type == "localization":
        args["label"] = label
        args["test_indist_only"] = test_indist_only
        label_data = dataset_class(**args)
        return Subset(label_data, list(range(len(label_data))))


class BalancedLoader:
    def __init__(
        self,
        normal_dataset,
        anomaly_dataset,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        if batch_size % 2 == 1:
            raise ValueError("batch_size must be even to ensure balancing")
        self.dl_normal = DataLoader(
            normal_dataset, batch_size // 2, shuffle=True, num_workers=num_workers // 2
        )
        self.dl_anomaly = DataLoader(
            anomaly_dataset,
            batch_size // 2,
            shuffle=True,
            num_workers=num_workers - num_workers // 2,
        )
        self.it_anomaly = iter(self.dl_anomaly)
        self.anomaly_backlog = None

    def __len__(self):
        return len(self.dl_normal)

    def __iter__(self):
        self.it_normal = iter(self.dl_normal)
        return self

    def __next__(self):
        # Note dataloaders are required to return a list of tensors
        batch_normal = next(self.it_normal)
        batch_length = len(batch_normal[0])

        # Underlying object state isn't mutated, only references are updated
        # -> No deepcopy required
        batch_anomaly = self.anomaly_backlog if self.anomaly_backlog else None

        while not batch_anomaly or len(batch_anomaly[0]) < batch_length:
            try:
                new_batch = next(self.it_anomaly)
                if batch_anomaly:
                    batch_anomaly = [
                        torch.cat((old_data, new_data))
                        for old_data, new_data in zip(batch_anomaly, new_batch)
                    ]
                else:
                    batch_anomaly = new_batch
            except StopIteration:
                self.it_anomaly = iter(self.dl_anomaly)

        if len(batch_anomaly[0]) == batch_length:
            self.anomaly_backlog = None
        else:
            self.anomaly_backlog = [
                batch_anomaly[batch_idx][batch_length:]
                for batch_idx in range(len(batch_normal))
            ]
            batch_anomaly = [
                batch_anomaly[batch_idx][:batch_length]
                for batch_idx in range(len(batch_normal))
            ]

        batch = [
            torch.cat((normal_data, anomalous_data))
            for normal_data, anomalous_data in zip(batch_normal, batch_anomaly)
        ]
        return batch


def mean_std(dataset: Subset, ae=False):
    tmp_transform = dataset.dataset.transform
    dataset.dataset.transform = None
    transforms = []
    if ae:
        transforms.append(v2.Grayscale())
    transforms += [v2.ToImage(), v2.ToDtype(torch.get_default_dtype(), scale=True)]
    transforms = v2.Compose(transforms)
    idx_list = range(len(dataset))
    try:
        if dataset.dataset.extend:
            idx_list = (
                np.argwhere(np.asarray(dataset.indices) < dataset.dataset.orig_len)
                .flatten()
                .tolist()
            )
    except AttributeError:
        pass
    imgs = [transforms(dataset[i][0]) for i in idx_list]
    dataset.dataset.transform = tmp_transform

    # Unpack into 1D channels to get around images in dataset being different sizes (ImageNet)
    n_ch = imgs[0].shape[-3]
    if n_ch != 1 and n_ch != 3:
        raise ValueError(f"Inputs must have 1 or 3 channels, got channels={n_ch}")
    ch_vals = [torch.cat([im[i, :, :].view(-1) for im in imgs]) for i in range(n_ch)]

    mean = [ch.mean() for ch in ch_vals]
    std = [ch.std() for ch in ch_vals]
    return mean, std
