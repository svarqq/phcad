from phcad.constants import SAVEROOT

DATADIR = SAVEROOT / "data"

FMNIST_LABELS = [
    "ankle-boot",
    "bag",
    "coat",
    "dress",
    "pullover",
    "sandal",
    "shirt",
    "sneaker",
    "top",
    "trouser",
]
FMNIST_LABEL_MAP = {lab: lab.capitalize().replace("-", " ") for lab in FMNIST_LABELS}
FMNIST_LABEL_MAP["top"] = "T-shirt/top"

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_LABEL_MAP = {lab: lab for lab in CIFAR10_LABELS}

MVTEC_DL_URL = (
    "https://www.mydrive.ch/"
    "shares/38536/3830184030e49fe74747669442f0f282/download/"
    "420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)
MVTEC_LABELS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
MVTEC_LABELS_NOFLIP = ["cable", "capsule", "metal_nut", "pill", "toothbrush"]
MVTEC_LABEL_MAP = {lab: lab for lab in MVTEC_LABELS}

MPDD_LABELS = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]
MPDD_LABEL_MAP = {lab: lab for lab in MPDD_LABELS}

OE_DATASET_MAP = {
    "cifar10": "cifar100",
    "fmnist": "cifar100",
    "mvtec": "imagenet21k-minus1k",
    "mpdd": "imagenet21k-minus1k",
}

DS_TO_LABELS_MAP = {
    "cifar10": CIFAR10_LABELS,
    "fmnist": FMNIST_LABELS,
    "mpdd": MPDD_LABELS,
    "mvtec": MVTEC_LABELS,
}
