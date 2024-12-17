import tomllib
import tarfile
import random
from pathlib import Path
from math import ceil
from functools import reduce, partial
from numbers import Number
from collections.abc import Collection
from collections import defaultdict

import requests
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.io as tio
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from phcad.types import AugmentableTransform
from phcad.data_handling.constants import DATADIR
from phcad.data_handling.augmentable_dataset import AugmentableDataset
from phcad.data_handling.noise_data import UniformNoiseImages


MVTEC_DATADIR = DATADIR / "mvtec-ad"
MVTEC_AD_URL = (
    "https://www.mydrive.ch/"
    "shares/38536/3830184030e49fe74747669442f0f282/download/"
    "420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)


def MVTec_AD(VisionDataset):
    # TODO: Basically everything haha
    pass


def mvtec_train_cal_dataloaders(
    label: str,
    resize_px: int,
    crop_px: int,
    train_size: int,
    cal_size: int,
):
    dpath, _, _ = generate_data(resize_px, crop_px)
    data = torch.load(dpath)[label]
    mean, std = (
        data.to(torch.get_default_dtype()).mean((0, 2, 3)),
        data.to(torch.get_default_dtype()).std((0, 2, 3)),
    )
    transform = partial(
        mvtec_transform,
        mean,
        std,
        data.shape[-3:],
        crop_px,
    )
    split_idx = (train_size * len(data)) // (train_size + cal_size)
    train_data = data[:split_idx]
    train_dset = AugmentableDataset(label, train_data, train_size, transform)
    trainloader = DataLoader(train_dset, batch_size=128, shuffle=True)

    if cal_size:
        cal_data = data[split_idx:]
        cal_pos = AugmentableDataset(label, cal_data, cal_size, transform)
        cal_neg = UniformNoiseImages(len(cal_pos), (3, crop_px, crop_px), mean, std)
        cal_dset = torch.utils.data.ConcatDataset([cal_pos, cal_neg])
        cal_loader = DataLoader(cal_dset, batch_size=128, shuffle=True)
    else:
        cal_loader = None

    return trainloader, cal_loader


def mvtec_transform(
    base_mean: tuple,
    base_std: tuple,
    shape_chw: tuple | None = None,
    crop_px: int | None = None,
    base: bool = True,
) -> AugmentableTransform:
    # For base preprocessing
    norm = transforms.Normalize(base_mean, base_std)
    transform_list = []
    if shape_chw[0] == 3:
        transform_list.append(transforms.Grayscale())
    if base:
        if crop_px:
            _, h, w = shape_chw
            crop = transforms.Lambda(
                lambda x: transforms.functional.crop(
                    x,
                    (h - crop_px) // 2,
                    (w - crop_px) // 2,
                    crop_px,
                    crop_px,
                )
            )
            transform_list.append(crop)
        transform_list.append(transforms.ToDtype(torch.get_default_dtype()))
        transform_list.append(norm)
        return transforms.Compose(transform_list)

    # For augmentation preprocessing
    gaussian_noise = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: x
                + torch.einsum("ijk, i -> ijk", torch.randn_like(x), base_std * 0.1)
            ),
            transforms.Lambda(lambda x: x.clamp(0, 255)),
        ]
    )

    jitter_choices = [
        transforms.ColorJitter(*[0.04 for _ in range(4)]),
        transforms.ColorJitter(*[0.0005 for _ in range(4)]),
    ]
    jitter_transform = random.choice(jitter_choices)
    transform_list.append(jitter_transform)
    transform_list.append(transforms.ToDtype(torch.double))
    if crop_px:
        random_crop = transforms.RandomCrop(crop_px)
        transform_list.append(random_crop)
    if random.random() >= 0.5:
        transform_list.append(gaussian_noise)
    transform_list.append(norm)
    return transforms.Compose(transform_list)


def unnormalize(img, mean, std):
    unnorm = transforms.Compose(
        [
            transforms.Normalize(torch.zeros(len(mean)), ([1 / st for st in std])),
            transforms.Normalize([-m for m in mean], torch.ones(len(std))),
        ]
    )
    return unnorm(img)


def generate_data(
    resize_px_train: int, resize_px_test: int, dataset_directory: Path = MVTEC_DATADIR
):
    trainpath, testpath, pxlabels_path = (
        dataset_directory / filename
        for filename in (
            f"train-{resize_px_train}px.pt",
            f"test-{resize_px_test}px.pt",
            "pxlabels.pt",
        )
    )
    train_generated, test_generated, pxlabels_generated = (
        p.exists() for p in [trainpath, testpath, pxlabels_path]
    )
    if train_generated and test_generated and pxlabels_generated:
        return trainpath, testpath, pxlabels_path

    def is_image(fname):
        img_exts = [".jpeg", ".jpg", ".png"]
        return any(
            map(
                lambda ext: fname.endswith(ext),
                img_exts + list(map(lambda ext: ext.upper(), img_exts)),
            )
        )

    # Load images from the extracted folder into tensors
    # Logic for test and pixel label data is designed so they are aligned by indices
    extract_path = download_and_extract_mvtec(dataset_directory)
    train_data, test_data, pxlabels = {}, {}, {}
    for dir, _, fnames in extract_path.walk():
        if not all(map(lambda fname: is_image(fname), fnames)):
            continue

        label, subset, defect_type = dir.parts[-3:]
        if subset == "ground_truth":
            continue
        elif subset == "train" and train_generated:
            continue
        elif subset == "test" and test_generated and pxlabels_generated:
            continue

        if subset == "train":
            imgs = map(
                lambda imgname: tio.read_image(
                    dir / imgname, mode=tio.ImageReadMode.RGB
                ),
                fnames,
            )
            if label not in train_data:
                train_data[label] = []
            train_data[label] += list(
                map(
                    lambda img: transforms.functional.resize(img, resize_px_train), imgs
                )
            )
            continue

        fnames.sort()
        if subset == "test" and not test_generated:
            imgs = map(
                lambda imgname: tio.read_image(
                    dir / imgname, mode=tio.ImageReadMode.RGB
                ),
                fnames,
            )
            if label not in test_data:
                test_data[label] = {}
            test_imgs = list(
                map(lambda img: transforms.functional.resize(img, resize_px_test), imgs)
            )
            test_data[label][defect_type] = test_imgs

        if subset == "test" and not pxlabels_generated:
            if defect_type != "good":
                gt_dir = Path(
                    "/" + "/".join(dir.parts[1:-2]) + "/ground_truth/" + dir.parts[-1]
                )
                masknames = [
                    fname.split(".")[0] + "_mask." + fname.split(".")[1]
                    for fname in fnames
                ]
                defect_pxlabels = list(
                    map(
                        lambda imgname: (~tio.read_image(gt_dir / imgname)[0]) // 255,
                        masknames,
                    )
                )
            else:
                test_img_shape = tio.read_image(dir / fnames[0]).shape
                defect_pxlabels = [
                    torch.ones(test_img_shape[-2:], dtype=torch.uint8)
                ] * len(fnames)

            if label not in pxlabels:
                pxlabels[label] = {}
            pxlabels[label][defect_type] = defect_pxlabels

    if not train_generated:
        for label, imgs in train_data.items():
            train_data[label] = torch.stack(imgs)
        torch.save(train_data, trainpath)

    if not test_generated:
        for label, imgs in test_data.items():
            test_data[label] = torch.cat(
                [
                    torch.stack(test_data[label][defect])
                    for defect in sorted(test_data[label].keys())
                ],
            )
        torch.save(test_data, testpath)

    if not pxlabels_generated:
        for label, imgs in pxlabels.items():
            pxlabels[label] = torch.cat(
                [
                    torch.stack(pxlabels[label][defect])
                    for defect in sorted(pxlabels[label].keys())
                ],
            )
        torch.save(pxlabels, pxlabels_path)

    return trainpath, testpath, pxlabels_path


def download_and_extract_mvtec(download_directory: Path = MVTEC_DATADIR) -> Path:
    extract_directory = download_directory / "extracted"

    with requests.head(MVTEC_AD_URL) as req:
        req.raise_for_status()
        filesize = int(req.headers["Content-Length"])
        archive_name = (
            req.headers["Content-Disposition"].split("filename=")[1].split(";")[0][1:-1]
        )
    filepath = download_directory / archive_name
    if (
        filepath.exists()
        and filepath.stat().st_size == filesize
        and extract_directory.exists()
    ):
        return extract_directory

    if not (filepath.exists() and filepath.stat().st_size == filesize):
        if not filepath.parent.exists():
            print(f"Making directory {filepath.parent}")
            filepath.parent.mkdir(parents=True)
        print(f"Downloading MVTec AD archive from {MVTEC_AD_URL} to {filepath}")
        with requests.get(MVTEC_AD_URL, stream=True) as req:
            with open(filepath, "wb") as archive:
                for chunk in (
                    pbar := tqdm(
                        req.iter_content(chunk_size=8192),
                        total=filesize,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    )
                ):
                    pbar.update(len(chunk))
                    archive.write(chunk)

    print(f"Extracting archive from {filepath} to {extract_directory}")
    with tarfile.open(filepath, "r") as archive:
        archive.extractall(path=extract_directory, filter="data")
    return filepath


if __name__ == "__main__":
    download_and_extract_mvtec()
