import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
import torchvision.transforms.v2.functional as F

from phcad.data_handling.constants import (
    DATADIR,
    FMNIST_LABEL_MAP,
    CIFAR10_LABEL_MAP,
    IMAGENET30_LABEL_MAP,
)
from phcad.data_handling.imagenet import ImageNet30

DATASET_MAP = {
    "fmnist": (FashionMNIST, FMNIST_LABEL_MAP, "classification"),
    "cifar10": (CIFAR10, CIFAR10_LABEL_MAP, "classification"),
    "cifar100": (CIFAR100, None, "classification"),
    "imagenet30": (ImageNet30, IMAGENET30_LABEL_MAP, "classification"),
}


def get_dataset(
    dataset_name: str,
    split: str,
    label: str | None = None,
    complement=False,
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
            "Cannot set parameter complement to True for train data, it is designed for onevall testing"
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

    if dataset_name == "imagenet30":
        parentdir = "imagenet1k"
    elif dataset_name == "imagenet21k-minus1k":
        parentdir = "imagenet21k"
    else:
        parentdir = dataset_name
    dataset_dir = datadir / parentdir

    args = {"root": dataset_dir}
    if dataset_name == "imagenet30":
        args["split"] = "train" if split == "train" else "val"
    else:
        args["train"] = True if split == "train" else False
        args["download"] = True
    full_data = dataset_class(**args)
    if not label:
        return Subset(full_data, list(range(len(full_data))))

    if dataset_type == "classification":
        labels = np.asarray(full_data.targets)
    elif dataset_type == "segmentation":
        labels = np.asarray(full_data.label_list)
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


def mean_std(dataset: Subset):
    tmp_transform = dataset.dataset.transform
    dataset.dataset.transform = None
    data = torch.stack(
        [
            F.to_dtype(
                F.to_image(dataset[i][0]), dtype=torch.get_default_dtype(), scale=True
            )
            for i in range(len(dataset))
        ]
    )
    dataset.dataset.transform = tmp_transform

    data = data.to(torch.get_default_dtype())
    num_channels = data.shape[1]
    if num_channels == 1:
        return (data.mean(),), (data.std(),)
    elif num_channels == 3:
        return data.mean((0, 2, 3)), data.std((0, 2, 3))
    else:
        raise ValueError(
            f"Inputs must have 1 or 3 channels, got channels={num_channels}"
        )
