import torch

eps = 1e-4


class PlattCal(torch.nn.Module):
    def __init__(self):
        super(PlattCal, self).__init__()
        self.temperature = torch.nn.Parameter(torch.empty(1))
        self.bias = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.temperature, mean=1, std=0.1)
        torch.nn.init.normal_(self.bias, mean=0, std=0.1)

    def forward(self, logits):
        return logits / self.temperature + self.bias


class PerPixelPlatt(torch.nn.Module):
    def __init__(self, wh_shape):
        super(PerPixelPlatt, self).__init__()
        self.temperature = torch.nn.Parameter(torch.empty(wh_shape))
        self.bias = torch.nn.Parameter(torch.empty(wh_shape))
        torch.nn.init.normal_(self.temperature, mean=1, std=0.1)
        torch.nn.init.normal_(self.bias, mean=0, std=0.1)

    def forward(self, logits):
        return torch.mul(logits, 1 / self.temperature) + self.bias


class BetaCal(torch.nn.Module):
    def __init__(self):
        super(BetaCal, self).__init__()
        self.a = torch.nn.Parameter(torch.empty(1))
        self.b = torch.nn.Parameter(torch.empty(1))
        self.c = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.a, mean=1, std=0.1)
        torch.nn.init.normal_(self.b, mean=1, std=0.1)
        torch.nn.init.normal_(self.c, mean=0, std=0.1)

    def forward(self, prob_estimates):
        self.a.data = torch.clamp(self.a, 0)
        self.b.data = torch.clamp(self.b, 0)
        prob_estimates = torch.clamp(prob_estimates, eps, 1 - eps)
        s1 = torch.log(prob_estimates)
        s2 = -torch.log(1 - prob_estimates)
        logits = self.a * s1 + self.b * s2 + self.c
        return logits


class PerPixelBeta(torch.nn.Module):
    def __init__(self, wh_shape):
        super(PerPixelBeta, self).__init__()
        self.a = torch.nn.Parameter(torch.empty(wh_shape))
        self.b = torch.nn.Parameter(torch.empty(wh_shape))
        self.c = torch.nn.Parameter(torch.empty(wh_shape))
        torch.nn.init.normal_(self.a, mean=1, std=0.1)
        torch.nn.init.normal_(self.b, mean=1, std=0.1)
        torch.nn.init.normal_(self.c, mean=0, std=0.1)

    def forward(self, prob_estimates):
        self.a.data = torch.clamp(self.a, 0)
        self.b.data = torch.clamp(self.b, 0)
        prob_estimates = torch.clamp(prob_estimates, eps, 1 - eps)
        s1 = torch.log(prob_estimates)
        s2 = -torch.log(1 - prob_estimates)
        logits = torch.mul(self.a, s1) + torch.mul(self.b, s2) + self.c
        return logits


class LinearActivation(torch.nn.Module):
    # Placeholder for ATS, little more than dummy wummy
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x
