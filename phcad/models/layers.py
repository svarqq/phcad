import torch


class PlattCal(torch.nn.Module):
    def __init__(self):
        super(PlattCal, self).__init__()
        self.temperature = torch.nn.Parameter(torch.empty(1))
        self.bias = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.temperature, mean=1, std=0.1)
        torch.nn.init.normal_(self.bias, mean=0, std=0.1)

    def forward(self, logits):
        return logits / self.temperature + self.bias


class PerPixelPlattCal(torch.nn.Module):
    def __init__(self, wh_shape):
        super(PerPixelPlattCal, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(wh_shape))
        self.bias = torch.nn.Parameter(torch.zeros(wh_shape))
        torch.nn.init.normal_(self.temperature, mead=1, std=0.1)
        torch.nn.init.normal_(self.bias, mean=0, std=0.1)

    def forward(self, logits):
        return torch.einsum("ijk, jk -> ijk", logits, 1 / self.temperature) + self.bias


class BetaCal(torch.nn.Module):
    eps = 1e-4

    def __init__(self):
        super(BetaCal, self).__init__()
        self.a = torch.nn.Parameter(torch.empty(1))
        self.b = torch.nn.Parameter(torch.empty(1))
        self.c = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.a, mean=1, std=0.1)
        torch.nn.init.normal_(self.b, mean=1, std=0.1)
        torch.nn.init.normal_(self.c, mean=0, std=0.1)

    def forward(self, prob_estimates):
        a, b = torch.clamp(self.a, 0), torch.clamp(self.b, 0)
        prob_estimates = torch.clamp(prob_estimates, BetaCal.eps, 1 - BetaCal.eps)
        s1 = torch.log(prob_estimates)
        s2 = -torch.log(1 - prob_estimates)
        logits = a * s1 + b * s2 + self.c
        return logits


class LinearActivation(torch.nn.Module):
    # Placeholder for ATS, little more than dummy wummy
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x
