from functools import partial

import torch
import torchvision.transforms.v2 as v2

label_to_zero = v2.Lambda(lambda x: torch.tensor(0, dtype=torch.get_default_dtype()))
label_to_one = v2.Lambda(lambda x: torch.tensor(1, dtype=torch.get_default_dtype()))


def synthetic_mask(img_wh, anomaly_targets=False):
    func = torch.ones if anomaly_targets else torch.zeros
    return v2.Lambda(lambda x: func(img_wh, dtype=torch.get_default_dtype()))


mask_to_class = v2.Lambda(
    lambda mask: torch.tensor(0, dtype=torch.get_default_dtype())
    if torch.all(mask == 0)
    else torch.tensor(1, dtype=torch.get_default_dtype())
)


def get_cifar_train_transform(mean, std, ae=False, **kwargs):
    transforms = [
        v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if not ae:
        transforms.append(v2.RandomCrop(32, padding=4))
    else:
        transforms.append(v2.Grayscale(1))
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        v2.Normalize(mean, std),
    ]
    transform = v2.Compose(transforms)
    return transform


def get_fmnist_train_transform(mean, std, ae=False, **kwargs):
    transforms = [
        v2.Grayscale(1),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    if ae:
        transforms.append(v2.CenterCrop(28))
    else:
        transforms.append(v2.RandomCrop(28, padding=3))
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        v2.Normalize(mean, std),
    ]
    transform = v2.Compose(transforms)
    return transform


def get_default_train_transform(
    mean, std, resize_px=256, crop_px=224, ae=False, flip=True, **kwargs
):
    transforms = [
        v2.Resize(resize_px),
        v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
    ]
    if ae:
        transforms.append(v2.Grayscale(1))
    if flip:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    transforms += [
        v2.RandomCrop(crop_px),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        v2.Normalize(mean, std),
    ]
    return v2.Compose(transforms)


def get_default_test_transform(
    mean, std, resize_px=256, crop_px=224, ae=False, **kwargs
):
    transforms = []
    if ae:
        transforms.append(v2.Grayscale(1))
    transforms += [
        v2.Resize(resize_px),
        v2.CenterCrop(crop_px),
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(mean, std),
    ]
    return v2.Compose(transforms)


def generic_norm_transform(mean, std, ae=False, gs=False, resize_px=None):
    transforms = []
    if ae or gs:
        transforms.append(v2.Grayscale(1))
    if resize_px:
        transforms.append(v2.Resize(resize_px))
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.get_default_dtype(), scale=True),
        v2.Normalize(mean, std),
    ]
    return v2.Compose(transforms)


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
    "mpdd": get_default_train_transform,
    "mpdd-ae": partial(get_default_train_transform, resize_px=292, crop_px=256),
    "mvtec": get_default_train_transform,
    "mvtec-ae": partial(get_default_train_transform, resize_px=292, crop_px=256),
}

TEST_TRANSFORM_MAP = {
    "cifar10": generic_norm_transform,
    "fmnist": generic_norm_transform,
    "mpdd": get_default_test_transform,
    "mpdd-ae": partial(get_default_test_transform, resize_px=292, crop_px=256),
    "mvtec": get_default_test_transform,
    "mvtec-ae": partial(get_default_test_transform, resize_px=292, crop_px=256),
}

SEG_TRAIN_TRANSFORM_MAP = {
    "bce": partial(get_default_train_transform, resize_px=292, crop_px=256),
    "ssim": partial(get_default_train_transform, resize_px=292, crop_px=256),
    "fcdd": get_default_train_transform,
}

SEG_TEST_TRANSFORM_MAP = {
    "bce": partial(get_default_test_transform, resize_px=256, crop_px=256),
    "ssim": partial(get_default_test_transform, resize_px=256, crop_px=256),
    "fcdd": partial(get_default_test_transform, resize_px=224, crop_px=224),
}
