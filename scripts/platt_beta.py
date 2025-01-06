import __init__  # noqa
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration

from phcad.trainers.calibrate import apply_posthoc_calibration


class TD(Dataset):
    def __init__(self, x, y):  # noqa
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    y_pos = torch.ones((15000,))
    y_neg = torch.zeros((15000,))
    y = torch.cat((y_pos, y_neg))

    p_pos = torch.rand((15000,)) * 0.7 + 0.1
    p_neg = torch.rand((15000,)) * 0.7
    p = torch.cat((p_pos, p_neg))
    p = torch.clamp(p, 1e-10, 1 - 1e-10)
    logits = torch.log(p / (1 - p))

    # logits_pos = torch.rand((15000,)) * 10000 + 30
    # logits_neg = torch.rand((15000,)) * 40
    # logits = torch.cat((logits_pos, logits_neg))

    lsc = logits

    ds = TD(logits, y)
    dl = DataLoader(ds, batch_size=128, num_workers=4)

    print("sklearn platt")
    clf = LogisticRegression(tol=1e-7)
    logs = np.array([[i] for i in lsc])
    clf.fit(logs, y.numpy())
    print(1 / clf.coef_[0][0], clf.intercept_[0])

    inputs_to_logits = model = lambda x: x
    pm = apply_posthoc_calibration("platt", inputs_to_logits, dl, num_scores=30000)
    print(list(pm.parameters()))

    print("sklearn beta")
    bc = BetaCalibration(parameters="abm")
    x = np.array([[i] for i in p])
    bc.fit(x, y)
    a, b, m = bc.calibrator_.map_
    print(a, b, -math.log(m**a / (1 - m) ** b))

    inputs_to_pests = lambda inputs: F.sigmoid(inputs_to_logits(inputs))
    bm = apply_posthoc_calibration("beta", inputs_to_pests, dl, num_scores=30000)
    print(list(bm.parameters()))

    print(F.binary_cross_entropy_with_logits(pm(lsc), y))
    print(
        F.binary_cross_entropy_with_logits(
            torch.from_numpy(clf.predict(lsc.reshape(-1, 1))).to(
                torch.get_default_dtype()
            ),
            y,
        )
    )
    print(F.binary_cross_entropy(bm(p), y))
    print(
        F.binary_cross_entropy(
            torch.from_numpy(bc.predict(p)).to(torch.get_default_dtype()), y
        )
    )
