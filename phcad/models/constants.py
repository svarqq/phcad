from phcad.models import cnn_fmnist, cnn_cifar10, wrn18, ae_mvtec, fcdd

resnet = wrn18.WideResNet18
autoencoder = ae_mvtec.AEMvTec
MODEL_MAP = {
    "fmnist": cnn_fmnist.CNN_FMNIST,
    "cifar10": cnn_cifar10.CNN_CIFAR10,
    "mpdd": resnet,
    "mvtec": resnet,
    "mpdd-ae": autoencoder,
    "mvtec-ae": autoencoder,
}

SEG_MODEL_MAP = {
    "bce": autoencoder(unet=True),
    "fcdd": fcdd.FCDD(),
    "ssim": autoencoder(),
}
