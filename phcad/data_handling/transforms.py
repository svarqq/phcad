import torch
import torchvision.transforms.v2 as v2

label_to_zero = v2.Lambda(lambda x: torch.tensor(0, dtype=torch.get_default_dtype()))
label_to_one = v2.Lambda(lambda x: torch.tensor(1, dtype=torch.get_default_dtype()))


def get_cifar_train_transform(mean, std, ae=False):
    transforms = [
        v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if not ae:
        transforms.append(v2.RandomCrop(32, padding=4))
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        v2.Normalize(mean, std),
    ]
    transform = v2.Compose(transforms)
    return transform


def get_fmnist_train_transform(mean, std, ae=False):
    transforms = [
        v2.Grayscale(1),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if not ae:
        transforms.append(v2.RandomCrop(28, padding=3))
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        v2.Normalize(mean, std),
    ]
    transform = v2.Compose(transforms)
    return transform


def generic_norm_transform(mean, std):
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.get_default_dtype(), scale=True),
            v2.Normalize(mean, std),
        ]
    )
    return transforms


def unnorm(im, mean, std):
    unnorm = v2.Compose(
        [
            v2.Normalize(torch.zeros(len(mean)), ([1 / st for st in std])),
            v2.Normalize([-m for m in mean], torch.ones(len(std))),
        ]
    )
    return unnorm(im)


TRAIN_TRANSFORM_MAP = {
    "cifar10": get_cifar_train_transform,
    "fmnist": get_fmnist_train_transform,
}

TEST_TRANSFORM_MAP = {
    "cifar10": generic_norm_transform,
    "fmnist": generic_norm_transform,
}
