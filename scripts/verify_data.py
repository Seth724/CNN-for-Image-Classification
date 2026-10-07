from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fashion_mnist_70_15_15.npz"
MANIFEST = ROOT / "data" / "processed" / "split_manifest.csv"
META = ROOT / "data" / "processed" / "metadata.json"


def main():
    for p in [DATA, MANIFEST, META]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required shared file: {p}")

    d = np.load(DATA)
    expected = {
        "x_train": (49000, 28, 28), "y_train": (49000,),
        "x_val": (10500, 28, 28), "y_val": (10500,),
        "x_test": (10500, 28, 28), "y_test": (10500,)
    }
    for k, shape in expected.items():
        if tuple(d[k].shape) != shape:
            raise RuntimeError(f"{k}: expected {shape}, got {d[k].shape}")

    manifest = pd.read_csv(MANIFEST)
    counts = manifest["split"].value_counts().to_dict()
    if counts != {"train": 49000, "val": 10500, "test": 10500}:
        raise RuntimeError(f"Manifest split counts are wrong: {counts}")

    for label_key, expected_per_class in [("y_train", 4900), ("y_val", 1050), ("y_test", 1050)]:
        c = np.bincount(d[label_key], minlength=10)
        if not np.all(c == expected_per_class):
            raise RuntimeError(f"Bad class balance in {label_key}: {c.tolist()}")

    with open(META, encoding="utf-8") as f:
        meta = json.load(f)

    print("Verification PASSED")
    print("Dataset :", meta["dataset"])
    print("Seed    :", meta["seed"])
    print("Train   :", d["x_train"].shape)
    print("Val     :", d["x_val"].shape)
    print("Test    :", d["x_test"].shape)


if __name__ == "__main__":
    main()
