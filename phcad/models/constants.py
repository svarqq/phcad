from phcad.models import cnn_fmnist, cnn_cifar10, wrn18, ae_mvtec

MODEL_MAP = {
    "fmnist": cnn_fmnist.CNN_FMNIST,
    "cifar10": cnn_cifar10.CNN_CIFAR10,
    "imagenet30": wrn18,
    "mpdd": wrn18,
    "mvtec": wrn18,
    "imagenet30-ae": ae_mvtec,
    "mpdd-ae": ae_mvtec,
    "mvtec-ae": ae_mvtec,
}
