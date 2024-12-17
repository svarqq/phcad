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
IMAGENET30_LABEL_MAP = {lab: (lab.replace("-", " "),) for lab in IMAGENET30_LABELS}
IMAGENET30_LABEL_MAP["american-alligator"] = (
    "American alligator",
    "Alligator mississipiensis",
)
IMAGENET30_LABEL_MAP["bikini"] = ("bikini", "two-piece")
IMAGENET30_LABEL_MAP["dial-telephone"] = ("dial telephone", "dial phone")
IMAGENET30_LABEL_MAP["dragonfly"] = (
    "dragonfly",
    "darning needle",
    "devil's darning needle",
    "sewing needle",
    "snake feeder",
    "snake doctor",
    "mosquito hawk",
    "skeeter hawk",
)
IMAGENET30_LABEL_MAP["grand-piano"] = ("grand piano", "grand")
IMAGENET30_LABEL_MAP["hotdog"] = ("hotdog", "hot dog", "red hot")
IMAGENET30_LABEL_MAP["revolver"] = ("revolver", "six-gun", "six-shooter")
IMAGENET30_LABEL_MAP["tank"] = (
    "tank",
    "army tank",
    "armored combat vehicle",
    "armoured combat vehicle",
)

OE_DATASET_MAP = {"cifar10": "cifar100", "fmnist": "cifar100"}
