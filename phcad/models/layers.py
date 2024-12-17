import torch


class PerPixelPlattScaling(torch.nn.Module):
    def __init__(self, wh_shape):
        super(PerPixelPlattScaling, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(wh_shape))
        self.bias = torch.nn.Parameter(torch.zeros(wh_shape))

    def forward(self, logits):
        return torch.einsum("ijk, jk -> ijk", logits, 1 / self.temperature) + self.bias


class LinearActivation(torch.nn.Module):
    # Placeholder for ATS, little more than dummy wummy
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x
