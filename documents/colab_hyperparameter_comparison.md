# Colab training and Model A/Model B comparison

This guide runs the existing training code in Google Colab, preserves every
configuration in Google Drive, and creates the comparison table required by
the assignment.

## 1. Clone the repository once

In Colab, the cloned repository is under `/content`, which is temporary to the
Colab runtime. The experiment outputs below are stored in Google Drive so they
survive a disconnect.

```python
from google.colab import drive
drive.mount("/content/drive")

import os
DRIVE_ROOT = "/content/drive/MyDrive/pattern_cnn_experiments"
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.environ["DRIVE_ROOT"] = DRIVE_ROOT
```

Run this clone cell only once per Colab runtime:

```python
%cd /content
!git clone https://github.com/Seth724/CNN-for-Image-Classification.git
%cd /content/CNN-for-Image-Classification
```

If the repository was already cloned, do not run `git clone` again. Use:

```python
%cd /content/CNN-for-Image-Classification
!git pull
```

The local repository path is:

```text
/content/CNN-for-Image-Classification
```

The persistent experiment path is:

```text
/content/drive/MyDrive/pattern_cnn_experiments
```

## 2. Check the GPU and data

Select **Runtime > Change runtime type > T4 GPU** before running training.
Do not run `scripts/prepare_data.py` if the processed file already exists.
That script needs the original Kaggle CSV files, but training only needs the
processed split.

```python
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
```

```python
!ls -lh data/processed
!python scripts/verify_data.py
```

Expected verification:

```text
Train   : (49000, 28, 28)
Val     : (10500, 28, 28)
Test    : (10500, 28, 28)
Verification PASSED
```

The raw CSV files are not required for this workflow. Only upload them if you
need to recreate the processed split:

```python
from google.colab import files
files.upload()
```

Then place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` in
`/content/CNN-for-Image-Classification/data/raw/` and run:

```python
!python scripts/prepare_data.py
!python scripts/verify_data.py
```

## 3. Run a smoke test

This checks imports, data loading, GPU execution, and output permissions:

```python
!python scripts/train_model_a.py \
  --epochs 1 \
  --batch-size 128 \
  --optimizer adam \
  --learning-rate 0.001 \
  --seed 42 \
  --output-dir "$DRIVE_ROOT/smoke_test"
```

## 4. Run Model A configurations

Every run must use a unique output directory. The script writes:

```text
<run>/models/model_a.keras
<run>/metrics/evaluation.json
<run>/metrics/history.csv
<run>/metrics/architecture.json
<run>/metrics/confusion_matrix.png
<run>/figures/training_curves.png
```

Run the initial optimizer comparison:

```python
!python scripts/train_model_a.py --epochs 20 --batch-size 128 --optimizer adam --learning-rate 0.001 --seed 42 --output-dir "$DRIVE_ROOT/model_a_adam_lr1e-3_bs128_e20"
!python scripts/train_model_a.py --epochs 20 --batch-size 128 --optimizer rmsprop --learning-rate 0.001 --seed 42 --output-dir "$DRIVE_ROOT/model_a_rmsprop_lr1e-3_bs128_e20"
!python scripts/train_model_a.py --epochs 20 --batch-size 128 --optimizer sgd --learning-rate 0.01 --seed 42 --output-dir "$DRIVE_ROOT/model_a_sgd_lr1e-2_bs128_e20"
!python scripts/train_model_a.py --epochs 20 --batch-size 128 --optimizer sgd_momentum --learning-rate 0.01 --momentum 0.9 --seed 42 --output-dir "$DRIVE_ROOT/model_a_sgd_momentum_lr1e-2_m09_bs128_e20"
```

After identifying the best optimizer, compare learning rates `0.0003`,
`0.001`, and `0.003`, then compare batch sizes `64`, `128`, and `256`.
Keep the seed and all other settings fixed during each comparison.

## 5. Run Model B

Model B saves to fixed paths inside the repository, so copy each run to Drive
before starting another Model B run:

```python
!python -m experiments.train_model_b --epochs 60 --batch-size 128 --lr 0.001 --seed 42
!python -m experiments.evaluate_model_b
!mkdir -p "$DRIVE_ROOT/model_b_adam_lr1e-3_bs128_e60"
!cp -r results/models "$DRIVE_ROOT/model_b_adam_lr1e-3_bs128_e60/"
!cp -r results/metrics "$DRIVE_ROOT/model_b_adam_lr1e-3_bs128_e60/"
!cp -r results/figures "$DRIVE_ROOT/model_b_adam_lr1e-3_bs128_e60/"
```

Model B's comparison metrics are in:

```text
results/metrics/model_b_metrics.json
```

## 6. Build the assignment comparison table

Run this after the Model A and Model B runs:

```python
import json
from pathlib import Path
import pandas as pd

root = Path("/content/drive/MyDrive/pattern_cnn_experiments")
rows = []

for path in root.glob("*/metrics/evaluation.json"):
    metrics = json.loads(path.read_text())
    rows.append({
        "model": "Model A",
        "run": path.parent.parent.name,
        "total_parameters": metrics.get("total_parameters"),
        "model_size_kb": metrics.get("model_size_kb"),
        "mean_epoch_time_s": metrics.get("mean_epoch_time_s"),
        "test_accuracy": metrics.get("test_accuracy"),
    })

for path in root.glob("*/metrics/model_b_metrics.json"):
    metrics = json.loads(path.read_text())
    rows.append({
        "model": "Model B",
        "run": path.parent.parent.name,
        "total_parameters": metrics.get("total_parameters"),
        "model_size_kb": metrics.get("model_size_kb"),
        "mean_epoch_time_s": metrics.get("mean_epoch_time_s"),
        "test_accuracy": metrics.get("test_accuracy"),
    })

comparison = pd.DataFrame(rows).sort_values(
    ["test_accuracy", "model"], ascending=[False, True]
)
display(comparison)
comparison.to_csv(root / "model_a_model_b_comparison.csv", index=False)
```

The required columns are:

- `total_parameters`
- `model_size_kb`
- `mean_epoch_time_s`
- `test_accuracy`

Model size is the actual saved `.keras` file size on disk, including its
container overhead. It is not the same as parameter count multiplied by four.

## 7. Interpreting the convolution trade-off

Model A uses standard convolutions. Model B uses depthwise separable
convolutions. Discuss the measured results rather than assuming one model is
better:

- Depthwise separable convolutions usually reduce parameters and computation,
  which can reduce model size and inference/training cost.
- Standard convolutions mix spatial and channel information in one operation
  and may provide stronger feature extraction, but generally use more
  parameters and computation.
- Compare accuracy against parameter count, saved-file size, and epoch time.
  A small accuracy difference may be worthwhile if Model B is substantially
  smaller or faster.
- Use the same dataset split and test protocol for both models.

## 8. Keep or delete runs

Do not delete a run until its `evaluation.json` or
`model_b_metrics.json` row has been copied into the comparison CSV. Keep the
best `.keras` model and its metrics/figures. Unsuccessful or duplicate runs
can then be removed from:

```text
/content/drive/MyDrive/pattern_cnn_experiments/
```
