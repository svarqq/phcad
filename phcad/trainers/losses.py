from functools import partial

import torch
from torch.nn.parameter import Buffer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from phcad.metrics import (
    ssim,
    hypersphere_metric,
    rbf_with_pseudo_huber,
)


class DSVDDLoss(_Loss):
    def __init__(self, center):
        super(DSVDDLoss, self).__init__()
        self.center = Buffer(center)

    def forward(self, model_outputs, **kwargs):
        f = partial(hypersphere_metric, center=self.center)
        return (torch.vmap(f)(model_outputs)).mean()

    def get_logits(self, model_outputs, **kwargs):
        f = partial(hypersphere_metric, center=self.center)
        return torch.vmap(f)(model_outputs)

    def get_pests(self, model_outputs, **kwargs):
        return F.sigmoid(self.get_logits(model_outputs))


class FCDDLoss(_Loss):
    def __init__(self):
        super(FCDDLoss, self).__init__()

    def forward(self, model_outputs, **kwargs):
        pass


class HSCLoss(_Loss):
    eps = 1e-4

    def __init__(self):
        super(HSCLoss, self).__init__()

    def forward(self, model_outputs, labels, **kwargs):
        p_estimates = self.get_pests(model_outputs)
        return F.binary_cross_entropy(p_estimates, labels)

    def get_logits(self, model_outputs, **kwargs):
        p = self.get_pests(model_outputs)
        logits = torch.log(p) - torch.log(1 - p)
        return logits

    def get_pests(self, model_outputs, **kwargs):
        p_estimates = torch.vmap(rbf_with_pseudo_huber)(model_outputs)
        return torch.clamp(p_estimates, HSCLoss.eps, 1 - HSCLoss.eps)


class CompositeBCE(_Loss):
    def __init__(self):
        super(CompositeBCE, self).__init__()

    def forward(self, model_outputs, labels, **kwargs):
        return F.binary_cross_entropy_with_logits(model_outputs, labels)

    def get_logits(self, model_outputs, **kwargs):
        return model_outputs

    def get_pests(self, model_outputs, **kwargs):
        return F.sigmoid(model_outputs)


class SSIMLoss(_Loss):
    eps = 1e-4

    # Note all SSIM values lie between [ims different <-- -1 and 1 --> ims identical]
    def __init__(self, reduce=True, **ssim_args):
        super(SSIMLoss, self).__init__()
        self.f = torch.vmap(partial(ssim, **ssim_args))
        self.ch_dim = -3
        self.px_dims = (-1, -2)
        self.reduce = reduce

    def forward(self, model_inputs, model_outputs, **kwargs):
        return 1 - (self.f(model_inputs, model_outputs, **kwargs)).mean()

    def get_logits(self, model_inputs, model_outputs, **kwargs):
        p = self.get_pests(model_inputs, model_outputs)
        logits = torch.log(p) - torch.log(1 - p)
        return logits

    def get_pests(self, model_inputs, model_outputs, **kwargs):
        p = (1 - self.f(model_inputs, model_outputs, **kwargs).mean(self.ch_dim)) / 2
        p = torch.clamp(p, SSIMLoss.eps, 1 - SSIMLoss.eps)
        if self.reduce:
            return p.mean(self.px_dims)
        else:
            return p


LOSS_MAP = {
    "bce": CompositeBCE(),
    "dsvdd": DSVDDLoss,
    "hsc": HSCLoss(),
    "ssim": SSIMLoss,
}

SEG_LOSS_MAP = {"bce": CompositeBCE(), "fcdd": None, "ssim": SSIMLoss}
