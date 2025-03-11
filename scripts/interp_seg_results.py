import json
import orjson
import numpy as np
from tqdm import tqdm
from pathlib import Path

d = Path("../results/").resolve()
fpr_pro = np.linspace(0, 0.3, 101)
fpr_roc = np.linspace(0, 1, 101)
for p, _, fs in d.walk():
    for f in tqdm(fs):
        print(f)
        with open(p / f) as f2:
            data = orjson.loads(f2.read())
        data["roc"]["y"] = np.interp(
            fpr_roc, data["roc"]["x"], data["roc"]["y"]
        ).tolist()
        data["roc"]["x"] = fpr_roc.tolist()
        data["pro"]["y"] = np.interp(
            fpr_pro, data["pro"]["x"], data["pro"]["y"]
        ).tolist()
        data["pro"]["x"] = fpr_pro.tolist()
        with open(f, "w") as f1:
            f1.write(json.dumps(data, indent=2))
