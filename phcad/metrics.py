import torch
import torch.nn.functional as F


def ssim(img1, img2, win_size=11, c1=0.01, c2=0.03, pad=False, **kwargs):
    bfac = win_size**2 / (win_size**2 - 1)  # factor for Bessel's correction of variance
    n_ch = img1.shape[0]
    window = torch.ones((n_ch, 1, win_size, win_size), requires_grad=True) / win_size**2
    if pad:
        # 0-pad -- Inputs are expected to be normalized about the training mean
        img1, img2 = (
            F.pad(img, (win_size // 2,) * 4, mode="constant") for img in (img1, img2)
        )

    m1, m2 = (F.conv2d(img, window, groups=n_ch) for img in (img1, img2))
    m1sq, m2sq, m12 = m1**2, m2**2, m1 * m2
    vx, vy = (
        bfac * (F.conv2d(img**2, window, groups=n_ch) - msq)
        for img, msq in zip((img1, img2), (m1sq, m2sq))
    )
    vxy = bfac * (F.conv2d(img1 * img2, window, groups=n_ch) - m12)
    ssim = ((2 * m1 * m2 + c1) * (2 * vxy + c2)) / (
        (m1**2 + m2**2 + c1) * (vx + vy + c2)
    )
    return ssim


def hypersphere_metric(features, center):
    return ((features - center) ** 2).sum()


def pseudo_huber_score(features, reduce=True):
    normsq = features**2
    if reduce:
        return torch.sqrt(normsq.sum() + 1) - 1
    else:
        return torch.sqrt(normsq + 1) - 1


def rbf_with_pseudo_huber(features):
    pseudo_huber = pseudo_huber_score(features)
    return torch.exp(-pseudo_huber)


def fcdd_anomaly_heatmap(features):
    return pseudo_huber_score(features, reduce=False)
