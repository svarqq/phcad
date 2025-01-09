from functools import partial

import torch
from torch.nn.parameter import Buffer
import torch.nn.functional as F

from phcad.metrics import hypersphere_metric, ssim
from phcad.trainers import losses


class BCEAnomalyScore(torch.nn.Module):
    def __init__(self):
        super(BCEAnomalyScore, self).__init__()

    def forward(self, model_outputs, **kwargs):
        return F.sigmoid(model_outputs)


class DSVDDAnomalyScore(torch.nn.Module):
    def __init__(self, center):
        super(DSVDDAnomalyScore, self).__init__()
        self.center = Buffer(center)

    def forward(self, model_outputs, **kwargs):
        f = partial(hypersphere_metric, center=self.center)
        return torch.vmap(f)(model_outputs)


class HSCAnomalyScore(torch.nn.Module):
    def __init__(self):
        super(HSCAnomalyScore, self).__init__()

    def forward(self, model_outputs, **kwargs):
        return losses.HSCLoss().get_pests(model_outputs)


class SSIMAnomalyScore(torch.nn.Module):
    def __init__(self, reduce=True, **ssim_args):
        super(SSIMAnomalyScore, self).__init__()
        self.f = torch.vmap(partial(ssim, **ssim_args))
        self.reduce = reduce

    def forward(self, model_inputs, model_outputs):
        ch_dim = -3
        px_dims = (-1, -2)
        ssim_vals = self.f(model_inputs, model_outputs).mean(ch_dim)
        if self.reduce:
            return 1 - ssim_vals.mean(px_dims)
        else:
            return 1 - ssim_vals


ANOMALY_SCORES = {
    "bce": BCEAnomalyScore(),
    "dsvdd": DSVDDAnomalyScore,
    "hsc": HSCAnomalyScore(),
    "ssim": SSIMAnomalyScore,
}

SEG_ANOMALY_SCORES = {"bce": None, "fcdd": None, "ssim": SSIMAnomalyScore}
