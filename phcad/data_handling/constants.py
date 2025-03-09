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

IMAGENET30_LABELS = [
    "acorn",
    "airliner",
    "ambulance",
    "american-alligator",
    "banjo",
    "barn",
    "bikini",
    "dial-telephone",
    "digital-clock",
    "dragonfly",
    "dumbbell",
    "forklift",
    "goblet",
    "grand-piano",
    "hotdog",
    "hourglass",
    "manhole-cover",
    "mosque",
    "nail",
    "parking-meter",
    "pillow",
    "revolver",
    "schooner",
    "snowmobile",
    "soccer-ball",
    "stingray",
    "strawberry",
    "tank",
    "toaster",
    "volcano",
]
IMAGENET30_LABEL_MAP = {lab: lab.replace("-", " ") for lab in IMAGENET30_LABELS}
IMAGENET30_LABEL_MAP["american-alligator"] = "American alligator"

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
MVTEC_LABELS_NOFLIP = ["cable", "capsul", "metal_nut", "pill", "toothbrush"]
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
    "imagenet30": "imagenet21k-minus1k",
    "mvtec": "cifar100",
    "mpdd": "cifar100",
}

DS_TO_LABELS_MAP = {
    "cifar10": CIFAR10_LABELS,
    "fmnist": FMNIST_LABELS,
    "imagenet30": IMAGENET30_LABELS,
    "mpdd": MPDD_LABELS,
    "mvtec": MVTEC_LABELS,
}
