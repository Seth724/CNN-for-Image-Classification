# CNN-for-Image-Classification

## Model A setup

From PowerShell, open the repository and activate the existing environment:

```powershell
cd "D:\7 sem\pattern CNN\CNN-for-Image-Classification"
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository expects these Kaggle Fashion-MNIST files in `data/raw/`:

```text
data/raw/fashion-mnist_train.csv
data/raw/fashion-mnist_test.csv
```

Prepare and verify the shared 70/15/15 split:

```powershell
python scripts/prepare_data.py
python scripts/verify_data.py
```

## Train Model A

Model A is the standard-convolution CNN in `src/models/model_a.py`. It is intentionally
uncompiled so each training script can choose its optimizer and metrics.

```powershell
python scripts/train_model_a.py --epochs 20 --batch-size 128
```

Outputs are written to `results/model_a/`:

- `results/model_a/models/model_a.keras`
- `results/model_a/metrics/architecture.json`
- `results/model_a/metrics/history.csv`
- `results/model_a/metrics/evaluation.json`
- `results/model_a/metrics/confusion_matrix.png`
- `results/model_a/figures/training_curves.png`

`evaluation.json` contains test loss and accuracy, macro precision, macro
recall, macro F1, weighted F1, per-class metrics, the confusion matrix, and
the measured training time. These are the results to use when comparing
models. The confusion matrix and training curves are suitable for the report.

Use fewer epochs for a quick smoke test:

```powershell
python scripts/train_model_a.py --epochs 1
```

## Optimizer comparison

The assignment requires comparing a selected optimizer with standard SGD and
SGD with momentum. Adam is a strong adaptive baseline, while RMSprop is an
optional extra comparison. Run each experiment in a separate output directory:

```powershell
python scripts/train_model_a.py --optimizer adam --epochs 20 --output-dir results/model_a_adam
python scripts/train_model_a.py --optimizer sgd --learning-rate 0.01 --epochs 20 --output-dir results/model_a_sgd
python scripts/train_model_a.py --optimizer sgd_momentum --learning-rate 0.01 --momentum 0.9 --epochs 20 --output-dir results/model_a_sgd_momentum
```

Optional RMSprop comparison:

```powershell
python scripts/train_model_a.py --optimizer rmsprop --epochs 20 --output-dir results/model_a_rmsprop
```

The script uses these defaults when `--learning-rate` is omitted: Adam and
RMSprop use `0.001`; SGD and SGD with momentum use `0.01`. These are reasonable
starting points, not universal best values. The learning rate controls the
step size of weight updates. Momentum `0.9` accumulates part of the previous
update, which usually reduces zig-zagging and speeds convergence, although an
excessive value can overshoot.

Compare `evaluation.json` files using the same dataset split, epoch count,
batch size, and random seed. Report test accuracy, macro F1, training time,
and the loss/accuracy curves. RMSprop may outperform Adam on this dataset;
the experiment should report the measured result rather than assume Adam wins.

Running a command again with the same `--output-dir` overwrites that directory's
model, history, figures, and metrics. Use a new output directory for every
configuration or repeat, for example `results/model_a_adam_seed43`.

## Model organization

Each architecture belongs in its own module under `src/models/`, for example
`src/models/model_a.py` and a future `src/models/model_b.py`. Export shared
builders from `src/models/__init__.py`; keep training and evaluation logic in
the scripts and utility modules rather than duplicating it inside model files.

The same repository can be cloned in Kaggle or Colab. Install the requirements,
make the dataset available in `data/raw/`, prepare the split, and run the same
training command. GPU-specific files do not need to be committed; commit the
scripts and metrics, while keeping large model files in notebook artifacts or
release storage.