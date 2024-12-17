import __init__
from phcad.models import cnn_fmnist, cnn_cifar10
import torch

model = cnn_cifar10.CNN_CIFAR10(ae=True, bias=False)
# print(list(model.layers.parameters()))
print("before", model)
model.prepare_calibration_network()
print("after", model)
rand = torch.rand((64, 3, 32, 32))
print(model(rand), model(rand).shape)
print(rand.shape)
