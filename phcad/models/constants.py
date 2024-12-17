from phcad.models import cnn_fmnist, cnn_cifar10

MODEL_MAP = {"fmnist": cnn_fmnist.CNN_FMNIST, "cifar10": cnn_cifar10.CNN_CIFAR10}
