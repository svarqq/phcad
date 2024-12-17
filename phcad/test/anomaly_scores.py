import torch.nn.functional as F


def probability_estimates(logits):
    return F.sigmoid(logits)
