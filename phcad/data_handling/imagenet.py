from pathlib import Path
from typing import Dict, List, Tuple, Union

from torchvision.datasets import ImageFolder
from torchvision.datasets.imagenet import load_meta_file


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
