import torch
from torch.nn.modules.loss import _Loss

from phcad.metrics import ssim


class SSIMLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model_inputs, model_outputs, **ssim_args):
        return 1 - (torch.vmap(ssim)(model_inputs, model_outputs, **ssim_args)).mean()
