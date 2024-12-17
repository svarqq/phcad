from pathlib import Path
from typing import Dict, List, Tuple, Union

from torchvision.datasets.imagenet import ImageNet, load_meta_file

from phcad.data_handling.constants import IMAGENET30_LABEL_MAP


class ImageNet30(ImageNet):
    label_map = {}

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
        return wnids, dict((wnid, i) for i, wnid in enumerate(wnids))
