from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
TRAIN_CSV = RAW_DIR / "fashion-mnist_train.csv"
TEST_CSV = RAW_DIR / "fashion-mnist_test.csv"
OUT_NPZ = OUT_DIR / "fashion_mnist_70_15_15.npz"
OUT_MANIFEST = OUT_DIR / "split_manifest.csv"
OUT_METADATA = OUT_DIR / "metadata.json"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csvs():
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(
            "Missing Kaggle CSV files. Put fashion-mnist_train.csv and "
            "fashion-mnist_test.csv inside data/raw/."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in [("train", train_df), ("test", test_df)]:
        if "label" not in df.columns:
            raise ValueError(f"{name} CSV has no 'label' column")
        if df.shape[1] != 785:
            raise ValueError(f"{name} CSV should have 785 columns, found {df.shape[1]}")

    x_train0 = train_df.drop(columns="label").to_numpy(dtype=np.uint8)
    y_train0 = train_df["label"].to_numpy(dtype=np.uint8)
    x_test0 = test_df.drop(columns="label").to_numpy(dtype=np.uint8)
    y_test0 = test_df["label"].to_numpy(dtype=np.uint8)

    x = np.concatenate([x_train0, x_test0]).reshape(-1, 28, 28)
    y = np.concatenate([y_train0, y_test0])

    ids = np.concatenate([
        np.array([f"original_train_{i:05d}" for i in range(len(y_train0))]),
        np.array([f"original_test_{i:05d}" for i in range(len(y_test0))])
    ])
    return x, y, ids


def split_by_class(y):
    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = [], [], []

    for c in range(10):
        idx = np.where(y == c)[0]
        if len(idx) != 7000:
            raise ValueError(f"Class {c} should contain 7000 samples, found {len(idx)}")
        idx = rng.permutation(idx)
        train_idx.extend(idx[:4900])
        val_idx.extend(idx[4900:5950])
        test_idx.extend(idx[5950:])

    train_idx = rng.permutation(np.asarray(train_idx, dtype=np.int64))
    val_idx = rng.permutation(np.asarray(val_idx, dtype=np.int64))
    test_idx = rng.permutation(np.asarray(test_idx, dtype=np.int64))
    return train_idx, val_idx, test_idx


def validate(y, train_idx, val_idx, test_idx):
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(all_idx) != 70000 or len(np.unique(all_idx)) != 70000:
        raise RuntimeError("Split overlap or missing samples detected")

    for name, idx, n, per_class in [
        ("train", train_idx, 49000, 4900),
        ("validation", val_idx, 10500, 1050),
        ("test", test_idx, 10500, 1050),
    ]:
        if len(idx) != n:
            raise RuntimeError(f"Wrong {name} size: {len(idx)}")
        counts = np.bincount(y[idx], minlength=10)
        if not np.all(counts == per_class):
            raise RuntimeError(f"Wrong {name} class counts: {counts.tolist()}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, sample_ids = load_csvs()
    if x.shape != (70000, 28, 28):
        raise ValueError(f"Unexpected full dataset shape: {x.shape}")

    train_idx, val_idx, test_idx = split_by_class(y)
    validate(y, train_idx, val_idx, test_idx)

    np.savez_compressed(
        OUT_NPZ,
        x_train=x[train_idx], y_train=y[train_idx],
        x_val=x[val_idx], y_val=y[val_idx],
        x_test=x[test_idx], y_test=y[test_idx],
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )

    split = np.empty(70000, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"

    pd.DataFrame({
        "global_index": np.arange(70000),
        "sample_id": sample_ids,
        "label": y,
        "class_name": [CLASS_NAMES[int(v)] for v in y],
        "split": split,
    }).to_csv(OUT_MANIFEST, index=False)

    metadata = {
        "dataset": "Fashion-MNIST",
        "source": "Zalando Research Fashion-MNIST Kaggle CSVs",
        "seed": SEED,
        "image_shape": [28, 28, 1],
        "total_samples": 70000,
        "num_classes": 10,
        "class_names": CLASS_NAMES,
        "splits": {
            "train": {"count": 49000, "ratio": 0.70, "per_class": 4900},
            "validation": {"count": 10500, "ratio": 0.15, "per_class": 1050},
            "test": {"count": 10500, "ratio": 0.15, "per_class": 1050}
        },
        "raw_file_sha256": {
            TRAIN_CSV.name: sha256(TRAIN_CSV),
            TEST_CSV.name: sha256(TEST_CSV)
        }
    }
    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done")
    print("Train      :", x[train_idx].shape)
    print("Validation :", x[val_idx].shape)
    print("Test       :", x[test_idx].shape)
    print("Saved      :", OUT_NPZ)


if __name__ == "__main__":
    main()
