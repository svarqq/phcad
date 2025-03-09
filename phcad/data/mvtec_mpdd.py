import logging
import tarfile
from PIL import Image
from pathlib import Path
import itertools

import requests
import tqdm
import numpy as np
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms.v2.functional as F

from phcad.data.constants import (
    MVTEC_DL_URL,
    MVTEC_LABEL_MAP,
    MPDD_LABEL_MAP,
)


logger = logging.getLogger(__name__)


class MVTecMPDD(VisionDataset):
    meta_fname = "meta.pt"
    train_extension_factor = 10

    def __init__(
        self,
        dataset_name,
        root,
        label=None,
        train=True,
        transform=None,
        target_transform=None,
        test_indist_only=False,
        **kwargs,
    ):
        if dataset_name != "mvtec" and dataset_name != "mpdd":
            raise ValueError(
                f'dataset_name must be one of ["mvtec", "mpdd"], got {dataset_name}'
            )
        super(MVTecMPDD, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train

        if dataset_name == "mvtec":
            extract_path = download_and_extract_mvtec(root)
            label_map = MVTEC_LABEL_MAP
        else:
            extract_path = root / "MPDD"
            label_map = MPDD_LABEL_MAP
        meta = self.generate_meta(self.root, extract_path)
        self.class_to_idx = meta["class_to_idx"]

        if train:
            orig_classes = meta["train_classes"]
        else:
            orig_classes = meta["test_classes"]

        if label:
            label_idcs = (
                np.argwhere(
                    np.asarray(orig_classes) == self.class_to_idx[label_map[label]]
                )
                .flatten()
                .tolist()
            )
        else:
            label_idcs = range(len(orig_classes))

        if train:
            label_data = [meta["train_data"][i] for i in label_idcs]
            label_classes = [orig_classes[i] for i in label_idcs]
            self.orig_len = len(label_data)

            self.extend = True
            self.data = list(
                itertools.chain.from_iterable(
                    [label_data] * MVTecMPDD.train_extension_factor
                )
            )
        else:
            all_data = [meta["test_data"][i] for i in label_idcs]
            if test_indist_only:
                indist_data = []
                for p, mask in all_data:
                    if not mask:
                        indist_data.append((p, mask))
                self.data = indist_data
            else:
                self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        impath, maskpath = self.data[idx]
        with open(impath, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if maskpath:
            with open(maskpath, "rb") as f:
                mask = Image.open(f)
                mask = mask.convert("L")
            mask = F.to_dtype(
                F.to_image(mask),
                dtype=torch.get_default_dtype(),
                scale=True,
            ).squeeze()
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
        else:
            mask = torch.zeros(img.size)  # In-distribution mask

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    @classmethod
    def generate_meta(cls, root, extract_dir):
        metapath = root / cls.meta_fname
        if metapath.exists():
            return torch.load(metapath, weights_only=False)

        img_exts = [".jpeg", ".jpg", ".png"]
        is_image = lambda fname: any(
            map(
                lambda ext: fname.endswith(ext),
                img_exts + [e.upper() for e in img_exts],
            )
        )

        train_data, test_data = [], []
        train_classes, test_classes = [], []
        class_to_idx, next_idx = {}, 0
        for dir, _, fnames in extract_dir.walk():
            if not fnames or not all(map(lambda fname: is_image(fname), fnames)):
                continue

            label, subset, defect_type = dir.parts[-3:]
            if label not in class_to_idx:
                class_to_idx[label] = next_idx
                next_idx += 1

            if subset == "train":
                train_classes += [class_to_idx[label]] * len(fnames)
                train_data += [(dir / fname, None) for fname in fnames]

            elif subset == "test":
                test_classes += [class_to_idx[label]] * len(fnames)
                mask_path = Path(
                    "/" + "/".join(dir.parts[1:-2]) + "/ground_truth/" + defect_type
                )
                if mask_path.exists():
                    test_data += [
                        (dir / fname, mask_path / fname.replace(".png", "_mask.png"))
                        for fname in fnames
                    ]
                else:
                    test_data += [(dir / fname, None) for fname in fnames]

        meta = {
            "train_data": train_data,
            "test_data": test_data,
            "train_classes": train_classes,
            "test_classes": test_classes,
            "class_to_idx": class_to_idx,
        }
        torch.save(meta, metapath)
        return meta


def download_and_extract_mvtec(download_directory):
    if not isinstance(download_directory, Path):
        download_directory = Path(download_directory)
    extract_directory = download_directory / "extracted"

    with requests.head(MVTEC_DL_URL) as req:
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
            logger.info(f"Making directory {filepath.parent}")
            filepath.parent.mkdir(parents=True)
        logger.info(f"Downloading archive from {MVTEC_DL_URL} to {filepath}")
        with requests.get(MVTEC_DL_URL, stream=True) as req:
            with open(filepath, "wb") as archive:
                for chunk in (
                    pbar := tqdm.tqdm(
                        req.iter_content(chunk_size=8192),
                        total=filesize,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    )
                ):
                    pbar.update(len(chunk))
                    archive.write(chunk)

    logger.info(f"Extracting archive from {filepath} to {extract_directory}")
    with tarfile.open(filepath, "r") as archive:
        archive.extractall(path=extract_directory, filter="data")
    return filepath
