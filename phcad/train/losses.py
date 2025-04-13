from functools import partial

import torch
from torch.nn.parameter import Buffer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from phcad.metrics import (
    ssim,
    hypersphere_metric,
    fcdd_anomaly_heatmap,
    rbf_with_pseudo_huber,
)


eps = 1e-4


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
        return torch.clamp(F.sigmoid(self.get_logits(model_outputs)), eps, 1 - eps)


class HSCLoss(_Loss):
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
        p_estimates = torch.clamp(
            1 - torch.vmap(rbf_with_pseudo_huber)(model_outputs), eps, 1 - eps
        )
        return p_estimates


class FCDDLoss(_Loss):
    def __init__(self, receptive_upsample_module):
        super(FCDDLoss, self).__init__()
        self.f = torch.vmap(fcdd_anomaly_heatmap)
        self.upsample = receptive_upsample_module

    def forward(self, model_outputs, labels, **kwargs):
        data_idcs = list(range(1, model_outputs.dim()))
        heatmaps = self.f(model_outputs).mean(data_idcs)
        pests_train = 1 - torch.exp(-heatmaps)
        return F.binary_cross_entropy(pests_train, labels)

    def get_logits(self, model_outputs, **kwargs):
        p = self.get_pests(model_outputs)
        logits = torch.log(p) - torch.log(1 - p)
        return logits

    def get_pests(self, model_outputs, **kwargs):
        upscaled_heatmaps = self.upsample(self.f(model_outputs)).squeeze(-3)
        pests = torch.clamp(1 - torch.exp(-upscaled_heatmaps), eps, 1 - eps)
        return pests


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
        p = torch.clamp(p, eps, 1 - eps)
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

SEG_LOSS_MAP = {"bce": CompositeBCE(), "fcdd": FCDDLoss, "ssim": SSIMLoss}
