from pathlib import Path
from typing import Dict, List, Tuple, Union

from torchvision.datasets import ImageFolder
from torchvision.datasets.imagenet import ImageNet, load_meta_file

from phcad.data_handling.constants import IMAGENET30_LABELS


class ImageNet30(ImageNet):
    raw_to_parsed_labels = {lab: (lab.replace("-", " "),) for lab in IMAGENET30_LABELS}
    raw_to_parsed_labels["american-alligator"] = (
        "American alligator",
        "Alligator mississipiensis",
    )
    raw_to_parsed_labels["bikini"] = ("bikini", "two-piece")
    raw_to_parsed_labels["dial-telephone"] = ("dial telephone", "dial phone")
    raw_to_parsed_labels["dragonfly"] = (
        "dragonfly",
        "darning needle",
        "devil's darning needle",
        "sewing needle",
        "snake feeder",
        "snake doctor",
        "mosquito hawk",
        "skeeter hawk",
    )
    raw_to_parsed_labels["grand-piano"] = ("grand piano", "grand")
    raw_to_parsed_labels["hotdog"] = ("hotdog", "hot dog", "red hot")
    raw_to_parsed_labels["revolver"] = ("revolver", "six-gun", "six-shooter")
    raw_to_parsed_labels["tank"] = (
        "tank",
        "army tank",
        "armored combat vehicle",
        "armoured combat vehicle",
    )

    def find_classes(self, directory):
        # Overriding impl of torchvision.datasets.folder.DatasetFolder
        if not isinstance(directory, Path):
            directory = Path(directory)
        rootdir = (directory / "..").resolve()
        wnids_to_labels = load_meta_file(rootdir)[0]
        labels_to_wnids = {v: k for k, v in wnids_to_labels.items()}
        conformed_labels = ImageNet30.raw_to_parsed_labels.values()
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
