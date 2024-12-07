import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from phcad.metrics import ssim


class SSIMLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model_inputs, model_outputs, **ssim_args):
        return 1 - (torch.vmap(ssim)(model_inputs, model_outputs, **ssim_args)).mean()


class CompositeBCE(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model_outputs, labels, **kwargs):
        return F.binary_cross_entropy_with_logits(model_outputs, labels)
