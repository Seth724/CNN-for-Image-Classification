# MobileNetV2 Fine-Tuning — Implementation Plan

**Branch:** `MobileNetV2_finetuning`
**Assignment parts covered:** Q5 (SOTA fine-tuning & evaluation) for MobileNetV2, and the MobileNetV2 inputs needed for Q6 (final comparison vs. Model B).
**Dataset:** Fashion-MNIST, shared 70/15/15 split (seed 42) in `data/processed/fashion_mnist_70_15_15.npz`.

---

## 1. Goal and constraints

| Requirement (from assignment) | How this plan meets it |
|---|---|
| Input resolution matches EfficientNet | Resize 28x28 to **32x32** so both SOTA models use the same input size |
| Same splits as custom networks | Read the shared `.npz` only; never re-split |
| Pre-trained lightweight model, fine-tuned | ImageNet MobileNetV2 (`alpha=1.0`), two-stage transfer learning |
| Test-set metrics | Accuracy, confusion matrix, per-class + macro precision/recall |
| Total parameter count and model size (MB) | Reported from the final saved `.keras` model |
| Q6 needs accuracy vs. memory vs. compute | Also record MACs/FLOPs, CPU latency per image, training time per epoch |

**Single-run rule:** one execution of the script must produce *every* number and figure needed for the report, so we never have to retrain. The test set is evaluated **once**, at the end.

---

## 2. Existing repo ? what we reuse

```
CNN-for-Image-Classification/
├── data/processed/fashion_mnist_70_15_15.npz   ← shared split (reuse, read-only)
├── data/processed/metadata.json                ← class names (reuse)
|   scripts/prepare_data.py, verify_data.py   | shared data tools |
|   src/data_loader.py                        | shared streaming loader |
```

- `src/data_loader.py::make_sota_datasets_streaming()` reads the shared split and resizes each batch to 32x32 RGB. Reuse it for MobileNetV2 without adding augmentation in this first run.
- Keep model-specific training in the MobileNetV2 implementation; save its report artifacts under `results/mobilenetv2/`.
- Shared loader changes stay additive so EfficientNet and MobileNetV2 can use the same streaming resize pipeline.

---

## 3. Files to add on this branch

```
documents/MobileNetV2_finetuning_plan.md   ← this file
MobileNetV2/model.py               ← full experiment (train → fine-tune → evaluate → report)
notebooks/colab_mobilenetv2.ipynb          ← thin Colab runner (mount Drive, clone, verify, run script)
results/mobilenetv2/                       ← small report artifacts copied back from Colab (json/csv/png)
```

Additive edits to shared files:
- `.gitignore`: model binaries (`*.keras`, `*.h5`, `*.tflite`) stay out of Git; report artifacts are saved under `results/mobilenetv2/`.

---

## 4. Model design

```
Input (32, 32, 3), raw pixels 0–255
 └─ Rescaling(1/127.5, offset=-1)          # = mobilenet_v2.preprocess_input, baked into the model
 └─ MobileNetV2(include_top=False, weights="imagenet", alpha=1.0)   # called with training=False
 ?? GlobalAveragePooling2D                  # 1x1x1280 -> 1280
 └─ Dropout(0.2)
 └─ Dense(10, softmax)
```

- **Preprocessing inside the model:** the saved/exported model then takes raw 0–255 images, which is simpler and safer to deploy on an edge device.
- **`training=False` on the base:** BatchNorm layers stay in inference mode in both stages. This is the standard fix for BN statistics getting wrecked during fine-tuning.
- **32x32 input:** this matches the current EfficientNet run and keeps the Q6 input-resolution comparison fair. MobileNetV2 downsamples by a factor of 32, so its final feature map is 1x1; this limits spatial detail for the classifier.
- **Expected size (approximate, verify from the run):** base ≈ 2.26 M params; head = 1280×10 + 10 = 12,810 → **≈ 2.27 M total**, about 8.7 MB as float32 weights.

---

## 5. Training procedure (one script run, two stages on the same model)

**Data pipeline:** use the shared 32x32 RGB datasets as returned by `make_tf_datasets`; do not add augmentation for this first run. Batch size 128, `prefetch(AUTOTUNE)`, seeds fixed (`tf.keras.utils.set_random_seed(42)`).

| | Stage 1 — feature extraction | Stage 2 — fine-tuning |
|---|---|---|
| Trainable | Head only (base frozen) | Last 30% of backbone layers + head; BN layers stay frozen |
| Optimizer | Adam, lr = 1e-3 | Adam, lr = **1e-5** (100× lower, so pretrained features are only nudged, not overwritten) |
| Epochs | up to 10 | up to 20 (the small LR converges more slowly, so it gets more epochs) |
| Callbacks | EpochTimer, ModelCheckpoint, EarlyStopping (patience 3, restore best weights) | EpochTimer, ModelCheckpoint, EarlyStopping (patience 5, restore best weights) |

- Stage 2 **continues from the Stage 1 weights**. It is not a second training from scratch.
- Hyperparameters are constants near the top of `MobileNetV2/model.py`. Set `MOBILENETV2_OUTPUT_DIR` in the environment to choose another results folder, such as Google Drive in Colab. The script currently has no CLI overrides or quick-training mode.
- The histories of both stages are concatenated so the loss/accuracy curves show one continuous timeline, with a vertical line marking where fine-tuning starts.
- Optimizer choice (justify in report): Adam reaches a good head quickly with default settings. The low LR in Stage 2 is what keeps fine-tuning stable. The Adam vs. SGD vs. SGD+momentum study (Q3) is done on the custom models, so it isn't repeated here.

---

## 6. Evaluation and outputs

The full run saves artifacts under `results/mobilenetv2/` by default. Set `MOBILENETV2_OUTPUT_DIR` to save them elsewhere, such as a Google Drive folder in Colab.

| File | Content | Assignment use |
|---|---|---|
| `metrics.json` | Test scores, parameter count, model size, FLOPs/MACs, and CPU latency | Q5 and Q6 |
| `confusion_matrix.csv` | Raw 10x10 confusion matrix | Q5 data/table |
| `classification_report.json` | Per-class precision, recall, and F1 | Q5 |
| `figures/final/confusion_matrix.png` | Test confusion matrix with raw counts | Q5 |
| `figures/final/per_class_metrics.png` | Test precision, recall, and F1 by class | Q5 |
| `figures/stage1/learning_curves.png`, `figures/stage2/learning_curves.png` | Train and validation curves for each stage | Training discussion |
| `figures/comparison/full_training_curve.png` | Train and validation curves across both stages | Training discussion |
| `stage1_history.json`, `stage2_history.json` | Train and validation history for each stage | Training discussion |
| `mobilenetv2_head_best.keras`, `mobilenetv2_finetuned_best.keras` | Validation-selected checkpoints | Recovery and reproducibility |
| `mobilenetv2_finetuned.keras` | Final fine-tuned model | Model size and future inference |

The full run selects weights and stops training using validation data. It predicts the full test split once after fine-tuning and derives test metrics and test plots from those predictions.

The `metrics.json` file includes the following fields:

```json
{
  "input_size": 32,
  "fine_tune_fraction": 0.3,
  "backbone_layers": 0,
  "selected_for_fine_tuning": 0,
  "trainable_backbone_layers": 0,
  "test_loss": 0.0,
  "test_accuracy": 0.0,
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f1": 0.0,
  "total_parameters": 0,
  "trainable_parameters": 0,
  "model_size_mb": 0.0,
  "flops_per_image": null,
  "estimated_macs_per_image": null,
  "cpu_inference_median_ms_per_image_batch1": null
}
```

How the measurements are produced:
- **Parameters:** Keras total parameter count plus trainable parameter counts after fine-tuning.
- **Model size:** bytes in the saved final `.keras` model divided by 1024 squared.
- **Compute:** TensorFlow profiler FLOPs per image, with MACs estimated as approximately half the FLOP count.
- **Timing:** epoch callback timings and median CPU inference latency for one synthetic 32x32 input, measured after 10 warmups and 100 timed runs. The synthetic input keeps latency measurement separate from the test set.
- TFLite export and quantization are optional edge-deployment extensions; the assignment does not require them.

---

## 7. Development workflow (local → GitHub → Colab)

### 7.1 Local (VS Code)
1. Confirm branch: `git switch MobileNetV2_finetuning`.
2. Write `MobileNetV2/model.py` following sections 4–6.
3. Check Python syntax with `python -m py_compile src/evaluation.py MobileNetV2/model.py`, then verify the shared split with `python scripts/verify_data.py`. The current script has no short-run or command-line override mode; running it starts both complete training stages.
4. Commit in small steps (the assignment checks commit history over time). Suggested commits:
   - `docs: add MobileNetV2 fine-tuning plan`
   - `feat: MobileNetV2 data pipeline with 32x32 RGB resize`
   - `feat: two-stage MobileNetV2 training (frozen head + fine-tune)`
   - `feat: evaluation, confusion matrix, precision/recall, curves`
   - `feat: model size, MACs and latency measurement`
   - `feat: Colab runner notebook`
   - `results: add MobileNetV2 metrics and figures from Colab run`
5. `git push -u origin MobileNetV2_finetuning`.

### 7.2 Google Colab (GPU run)
Notebook settings: **Runtime → Change runtime type → T4 GPU**.

Colab runtime files under `/content` are temporary, so set `MOBILENETV2_OUTPUT_DIR` to a folder in Google Drive before launching the script. The output path is read from this environment variable; the script does not currently accept `--output_dir`.

```python
# Cell 1 — mount Google Drive (outputs survive a disconnect)
from google.colab import drive
drive.mount('/content/drive')
OUT = "/content/drive/MyDrive/EN3150_A03/mobilenetv2"
import os
os.environ["MOBILENETV2_OUTPUT_DIR"] = OUT

# Cell 2 — get the code + shared data
!git clone --branch MobileNetV2_finetuning --single-branch https://github.com/Seth724/CNN-for-Image-Classification.git
%cd /content/CNN-for-Image-Classification

# Cell 3 — environment check (do NOT pip install requirements.txt:
#          reinstalling tensorflow on Colab can break its CUDA setup)
import tensorflow as tf, matplotlib
print(tf.__version__, tf.config.list_physical_devices("GPU"))

# Cell 4 — confirm the shared split
!python scripts/verify_data.py

# Cell 5 — syntax check (this does not start training)
!python -m py_compile src/evaluation.py MobileNetV2/model.py

# Cell 6 — full run (trains the head, fine-tunes, then evaluates once)
!python MobileNetV2/model.py

# Cell 7 — inspect saved results
!ls -lah "{OUT}"
```

- If you pushed new commits after cloning, run `!git pull` inside the repo folder before rerunning.
- Colab sessions can end, so saving to Drive preserves completed artifacts. The current script saves best checkpoints but does not automatically resume a stopped training run; rerunning starts training again.

### 7.3 Back to the repo
1. Copy the small artifacts (`*.json`, `*.csv`, `*.png`) from Google Drive into `results/mobilenetv2/` if you want them tracked in the repository. Leave out model binaries unless the team explicitly wants to store them.
2. Commit and push, then open a PR into `main`.

---

## 8. Team coordination checklist

- [ ] Agree on shared comparison columns for Model B and the two SOTA models so the Q6 table uses consistent definitions.
- [ ] Second SOTA owner (e.g. EfficientNet-B0) uses the **same 32x32 input and data split**, so the comparison is fair.
- [ ] Agree that "model size" means the **float32 file on disk** (KB for custom models, MB for SOTA), with TFLite sizes as an extra column.
- [ ] Model B owner shares parameter count, model size, compute estimate, and test metrics for the Q6 comparison.

---

## 9. Report content this run feeds (Q5 / Q6)

**Q5 (MobileNetV2 section):**
- Why MobileNetV2: inverted residuals + linear bottlenecks + depthwise separable convs; ReLU6 is cheap and quantization-friendly.
- Setup table: input, frozen/unfrozen layers, LR per stage, and epochs actually run.
- Figures: training curves, confusion matrix.
- Table: accuracy, macro precision/recall, total params, size (MB).
- Interpretation: which classes get confused (expect Shirt ↔ T-shirt/top ↔ Pullover ↔ Coat), and the frozen vs. fine-tuned accuracy jump (domain gap between ImageNet colour photos and 28×28 grayscale clothing).

**Q6 (vs. Model B):**
- Accuracy vs. params (~2.27 M vs. <100 k ≈ 20×+ more) vs. size (MB vs. KB) vs. MACs vs. CPU latency.
- Trade-offs: pretrained features and faster convergence vs. a much bigger memory footprint. At 32x32, MobileNetV2 ends with a 1x1 feature map, so the classifier receives little spatial detail. A from-scratch model can be fitted to an MCU memory budget, while MobileNetV2 usually needs Raspberry-Pi-class hardware unless it is quantized or `alpha` is reduced.
- Optional extra (if time allows): a run with `alpha=0.35` (~0.4 M params) to show the width-multiplier knob.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Colab disconnects mid-run | Outputs and best checkpoints are written to Google Drive; the script does not automatically resume a stopped training run |
| Stage 2 at lr 1e-5 still improving when epochs run out | Check the curves; if val loss is still falling at epoch 20, rerun with `--epochs2 30` (or `--lr2 3e-5`) |
| Need to "train twice" | Script writes every artifact in one run; test evaluation happens once at the end |
| Fine-tuning destabilises / overfits | Low LR, BN in inference mode, and EarlyStopping with best-weight restore |
| GPU/RAM OOM from full-array resize | Per-batch resize in `tf.data`; batch 128 |
| Colab environment changes | Use the installed TensorFlow and matplotlib; do not reinstall TensorFlow |
| Merge conflicts with teammates | Only additive edits to shared files; new code lives in new files |
| Low spatial detail at 32x32 | Report that the final feature map is 1x1 and discuss the resolution trade-off |
