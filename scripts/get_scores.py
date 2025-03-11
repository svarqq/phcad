import __init__

import torch
import torchvision.transforms.v2 as transforms
from sklearn import metrics
import matplotlib.pyplot as plt

from phcad.models.ae_mvtec import AEMvTec
from phcad.train.constants import CHKPTDIR
from phcad.data import mvtec_dataset
from phcad.metrics import ssim


def get_scores_aemvtec():
    labels = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    train_path, test_path, labelpath = mvtec_dataset.generate_data(276, 256)
    for label in labels:
        traindata = torch.load(train_path)[label]
        mean, std = (
            traindata.to(torch.get_default_dtype()).mean((0, 2, 3)),
            traindata.to(torch.get_default_dtype()).std((0, 2, 3)),
        )

        testdata = torch.load(test_path)[label]
        if testdata.shape[1] == 3:
            gs = transforms.Grayscale()
            testdata = gs(testdata)
        norm = transforms.Normalize(mean, std)
        testdata = norm(testdata.to(torch.double))

        gtmasks = torch.load(labelpath)[label]

        for seed in range(1, 2):
            for nlayers in range(0, 4):
                loaddir_cal = (
                    CHKPTDIR
                    / "mvtec-ae"
                    / "checkpoints"
                    / f"mvtec-ae-cal-{label}-nl{nlayers}-{seed}.pt"
                )
                model = AEMvTec()
                model.setup_cal((256, 256), nlayers)
                model_state = torch.load(loaddir_cal, map_location="cpu")["model_state"]
                model.load_state_dict(model_state)
                model.requires_grad_(False)
                get_scores_p(model, testdata, gtmasks)


def get_scores_p(model, test_data, gtmasks):
    gtmasks = gtmasks
    outputs = model(test_data)

    link = torch.nn.Sigmoid()
    prob_scores = link(outputs)
    interp_prob_scores = torch.nn.functional.interpolate(
        prob_scores.unsqueeze(1), gtmasks.shape[-1], mode="bicubic"
    )[:, 0, :, :]
    print("test loss", interp_prob_scores.shape, gtmasks.shape)
    roc = metrics.roc_auc_score(
        gtmasks.flatten().detach().numpy(),
        interp_prob_scores.flatten().detach().numpy(),
    )
    print("auroc", roc)

    entropy_scores = -prob_scores * torch.log2(prob_scores) - (
        1 - prob_scores
    ) * torch.log2(1 - prob_scores)
    entropy_scores = torch.nn.functional.interpolate(
        entropy_scores.unsqueeze(1), gtmasks.shape[-1], mode="bicubic"
    )[:, 0, :, :]
    roc2 = metrics.roc_auc_score(
        gtmasks.flatten().detach().numpy(), entropy_scores.flatten().detach().numpy()
    )
    print("entropy auroc", roc2)


def get_scores_ssim(model, test_data, gtmasks, mean=0, std=0):
    gtmasks = gtmasks
    outputs = model(test_data)

    plt.imshow(
        mvtec_dataset.unnormalize(test_data[0], mean, std)
        .to(torch.uint8)
        .permute((1, 2, 0))
    )
    plt.show()
    plt.imshow(
        mvtec_dataset.unnormalize(outputs[0], mean, std)
        .to(torch.uint8)
        .permute((1, 2, 0))
    )
    plt.show()
    plt.imshow(gtmasks[0])
    plt.show()

    scores = (1 - torch.vmap(ssim)(test_data, outputs, pad=True)).mean(1).unsqueeze(1)
    print(scores.mean())
    scores = torch.nn.functional.interpolate(scores, gtmasks.shape[-1], mode="bicubic")[
        :, 0, :, :
    ]
    print("test loss", scores.shape, gtmasks.shape)
    roc = metrics.roc_auc_score(
        gtmasks.flatten().detach().numpy(), scores.flatten().detach().numpy()
    )
    print("auroc", roc)

    pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    get_scores_aemvtec()
