from pathlib import Path
from typing import Dict, List, Tuple, Union

from torchvision.datasets import ImageFolder
from torchvision.datasets.imagenet import ImageNet, load_meta_file

from phcad.data_handling.constants import IMAGENET30_LABEL_MAP


class ImageNet30(ImageNet):
    def find_classes(
        self, directory: Union[str, Path]
    ) -> Tuple[List[str], Dict[str, int]]:
        # Overriding impl of torchvision.datasets.folder.DatasetFolder
        if not isinstance(directory, Path):
            directory = Path(directory)
        rootdir = (directory / "..").resolve()
        wnids_to_labels = load_meta_file(rootdir)[0]
        labels_to_wnids = {v: k for k, v in wnids_to_labels.items()}
        conformed_labels = IMAGENET30_LABEL_MAP.values()
        wnids = [labels_to_wnids[label] for label in conformed_labels]
        return wnids, {wnid: i for i, wnid in enumerate(wnids)}


class ImageNet21KMinus1K(ImageFolder):
    in21k_folder = "winter21_whole"

    def find_classes(
        self, directory: Union[str, Path]
    ) -> Tuple[List[str], Dict[str, int]]:
        if not isinstance(directory, Path):
            directory = Path(directory)
        with open(directory / "wnid-folders.txt") as f:
            wnid_folders = f.read().split("\n")
        while not wnid_folders[0]:
            wnid_folders = wnid_folders[1:]
        while not wnid_folders[-1]:
            wnid_folders = wnid_folders[:-1]

        wnid_folders = set([Path(folder).parts[1] for folder in wnid_folders])
        in1k_wnids = set(load_meta_file(directory)[0].keys())
        oe_wnids = list(wnid_folders - in1k_wnids)

        self.root = directory / ImageNet21KMinus1K.in21k_folder

        return oe_wnids, {oe_wnid: i for i, oe_wnid in enumerate(oe_wnids)}
