# MobileNetV2 Experiment Run Notes

**Code update after Run 7:** Each model keeps its own training stages inside `model.py`. Experiments use validation data; final testing runs through that same file with `--test_with_test_data "RUN_DIR"`. The new 30% rule selects 46 non-BN MobileNet backbone layers, replacing the old rule that left 30 trainable after selecting a 46-layer window. Runs 1-7 below retain their original settings and results; none was produced using the new quota rule. Future new-code runs are new experiments, rather than reproductions of Run 4's reported accuracy.

This file records the MobileNetV2 runs shared in the chat for later use in the assignment report. Model choices are compared using **validation data**. The test set is reserved for final reporting and must not be used to choose hyperparameters.

## Historical shared configuration (Runs 1-7)

| Setting | Value |
|---|---:|
| Dataset | Fashion-MNIST, shared 70/15/15 split |
| Input size | 32x32 for Runs 1-3; 64x64 for Runs 4-7; grayscale inputs resized and converted to RGB by the SOTA loader |
| Batch size | 32 for Runs 1 and 2; 8 for Runs 3-7 |
| Seed | 42 |
| Stage 1 | 10 epochs maximum; classifier head only; learning rate 1e-3 |
| Stage 2 | Fine-tune last 30% in Runs 1-4; last 20% in Runs 5-7; learning rate 1e-5 |
| Backbone layers | 154 total; 30% selects 46 layers (30 trainable after freezing BatchNormalization); 20% selects 30 layers (19 trainable after freezing BatchNormalization) |
| Augmentation | None |
| Dropout | 0.2 for Runs 1-5; 0.3 for Runs 6-7 |
| Selection | Best validation loss; early stopping restores best weights |

## Run 1: Stage 2 limit of 20 epochs

Stage 1 ended with validation loss **1.5762** and validation accuracy **42.09%**. Stage 2 reached its 20-epoch limit. Its best/final validation loss was **0.3551** and validation accuracy was **87.75%**.

| Stage | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| Head | 1 | 1.8653 | 35.19% | 1.7089 | 39.59% |
| Head | 2 | 1.7050 | 38.07% | 1.6463 | 40.45% |
| Head | 3 | 1.6716 | 38.61% | 1.6208 | 40.98% |
| Head | 4 | 1.6550 | 39.18% | 1.6065 | 41.24% |
| Head | 5 | 1.6448 | 39.22% | 1.5975 | 41.65% |
| Head | 6 | 1.6385 | 39.40% | 1.5899 | 41.75% |
| Head | 7 | 1.6324 | 39.57% | 1.5868 | 41.76% |
| Head | 8 | 1.6291 | 39.63% | 1.5828 | 41.94% |
| Head | 9 | 1.6253 | 39.77% | 1.5794 | 42.12% |
| Head | 10 | 1.6228 | 39.90% | 1.5762 | 42.09% |
| Fine-tune | 1 | 0.9544 | 66.93% | 0.6678 | 76.60% |
| Fine-tune | 2 | 0.6613 | 76.61% | 0.5420 | 81.25% |
| Fine-tune | 3 | 0.5721 | 79.61% | 0.5123 | 81.46% |
| Fine-tune | 4 | 0.5156 | 81.53% | 0.4653 | 83.53% |
| Fine-tune | 5 | 0.4806 | 82.66% | 0.4423 | 84.30% |
| Fine-tune | 6 | 0.4549 | 83.67% | 0.4406 | 84.50% |
| Fine-tune | 7 | 0.4338 | 84.31% | 0.4311 | 84.92% |
| Fine-tune | 8 | 0.4102 | 85.23% | 0.4021 | 85.51% |
| Fine-tune | 9 | 0.3926 | 85.74% | 0.3999 | 85.52% |
| Fine-tune | 10 | 0.3787 | 86.22% | 0.4069 | 85.79% |
| Fine-tune | 11 | 0.3679 | 86.67% | 0.4077 | 85.71% |
| Fine-tune | 12 | 0.3576 | 86.99% | 0.3845 | 86.61% |
| Fine-tune | 13 | 0.3426 | 87.33% | 0.3895 | 86.63% |
| Fine-tune | 14 | 0.3341 | 87.75% | 0.3784 | 86.43% |
| Fine-tune | 15 | 0.3234 | 88.04% | 0.3834 | 86.48% |
| Fine-tune | 16 | 0.3162 | 88.33% | 0.3610 | 87.15% |
| Fine-tune | 17 | 0.3030 | 88.73% | 0.3733 | 87.18% |
| Fine-tune | 18 | 0.3008 | 88.83% | 0.3577 | 87.53% |
| Fine-tune | 19 | 0.2931 | 88.93% | 0.3672 | 87.26% |
| Fine-tune | 20 | 0.2825 | 89.35% | 0.3551 | 87.75% |

Reported final test metrics for this run (for final reporting, not model selection): test loss **0.3404**, accuracy **88.20%**, macro precision **0.8823**, macro recall **0.8820**, macro F1 **0.8811**. Computation estimates: **12,636,668 FLOPs/image**, **6,318,334 estimated MACs/image**. Median CPU latency, batch size 1: **3.3242 ms/image**. The script saved the run's plots and metrics to the configured Drive output directory.

## Run 2: Stage 2 limit of 30 epochs

This run repeated the same Stage 1 and the first 20 Stage 2 epochs, then continued Stage 2 to epoch 30. Best validation loss improved to **0.3456** at epoch 28. Best validation accuracy was **88.01%** at epoch 26. The last epoch ended at train accuracy **91.31%**, validation accuracy **87.85%**, and validation loss **0.3667**.

| Fine-tune epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.9544 | 66.93% | 0.6678 | 76.60% |
| 2 | 0.6613 | 76.61% | 0.5420 | 81.25% |
| 3 | 0.5721 | 79.61% | 0.5123 | 81.46% |
| 4 | 0.5156 | 81.53% | 0.4653 | 83.53% |
| 5 | 0.4806 | 82.66% | 0.4423 | 84.30% |
| 6 | 0.4549 | 83.67% | 0.4406 | 84.50% |
| 7 | 0.4338 | 84.31% | 0.4311 | 84.92% |
| 8 | 0.4102 | 85.23% | 0.4021 | 85.51% |
| 9 | 0.3926 | 85.74% | 0.3999 | 85.52% |
| 10 | 0.3787 | 86.22% | 0.4069 | 85.79% |
| 11 | 0.3679 | 86.67% | 0.4077 | 85.71% |
| 12 | 0.3576 | 86.99% | 0.3845 | 86.61% |
| 13 | 0.3426 | 87.33% | 0.3895 | 86.63% |
| 14 | 0.3341 | 87.75% | 0.3784 | 86.43% |
| 15 | 0.3234 | 88.04% | 0.3834 | 86.48% |
| 16 | 0.3162 | 88.33% | 0.3610 | 87.15% |
| 17 | 0.3030 | 88.73% | 0.3733 | 87.18% |
| 18 | 0.3008 | 88.83% | 0.3577 | 87.53% |
| 19 | 0.2931 | 88.93% | 0.3672 | 87.26% |
| 20 | 0.2825 | 89.35% | 0.3551 | 87.75% |
| 21 | 0.2810 | 89.58% | 0.3596 | 87.73% |
| 22 | 0.2699 | 89.91% | 0.3617 | 87.29% |
| 23 | 0.2647 | 90.17% | 0.3715 | 86.58% |
| 24 | 0.2583 | 90.37% | 0.3460 | 87.90% |
| 25 | 0.2518 | 90.55% | 0.3611 | 87.59% |
| 26 | 0.2491 | 90.60% | 0.3559 | 88.01% |
| 27 | 0.2430 | 90.87% | 0.3715 | 86.85% |
| 28 | 0.2382 | 90.91% | **0.3456** | 87.70% |
| 29 | 0.2315 | 91.16% | 0.3771 | 87.99% |
| 30 | 0.2286 | 91.31% | 0.3667 | 87.85% |

Other logged measurements: **12,636,668 FLOPs/image**, **6,318,334 estimated MACs/image**, and median CPU latency **2.9250 ms/image** at batch size 1. The pasted log ended at "Final evaluation on the test set"; no final test metrics for this run were included in the pasted report. Do not infer them from Run 1.

## Run 3: batch size 8, Stage 2 limit of 30 epochs

The log shows **6,125 batches per epoch** for 49,000 training images, confirming batch size 8. The model still selected the last 46 of 154 backbone layers (30%); BatchNormalization remained frozen, leaving 30 selected backbone layers trainable. Stage 2 stopped after epoch 23 because the best validation loss was at epoch 18 and did not improve for the next five epochs. Best validation loss was **0.3245** (epoch 18); best validation accuracy was **88.87%** (epoch 22). This is a batch-size-8 run, not a fine-tuning-fraction experiment.

| Stage | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| Head | 1 | 1.7756 | 36.51% | 1.6392 | 40.57% |
| Head | 2 | 1.6624 | 38.80% | 1.6041 | 41.15% |
| Head | 3 | 1.6418 | 39.29% | 1.5905 | 41.63% |
| Head | 4 | 1.6320 | 39.67% | 1.5803 | 41.97% |
| Head | 5 | 1.6270 | 39.70% | 1.5776 | 42.04% |
| Head | 6 | 1.6230 | 39.95% | 1.5720 | 42.23% |
| Head | 7 | 1.6193 | 40.01% | 1.5711 | 42.42% |
| Head | 8 | 1.6168 | 39.97% | 1.5695 | 42.50% |
| Head | 9 | 1.6156 | 40.07% | 1.5682 | 42.58% |
| Head | 10 | 1.6138 | 40.12% | 1.5677 | 42.42% |
| Fine-tune | 1 | 0.9426 | 66.69% | 0.5690 | 78.93% |
| Fine-tune | 2 | 0.6402 | 77.53% | 0.5005 | 81.92% |
| Fine-tune | 3 | 0.5434 | 81.00% | 0.4744 | 83.05% |
| Fine-tune | 4 | 0.4899 | 82.68% | 0.4223 | 84.80% |
| Fine-tune | 5 | 0.4515 | 84.18% | 0.4071 | 85.29% |
| Fine-tune | 6 | 0.4231 | 85.01% | 0.4041 | 85.96% |
| Fine-tune | 7 | 0.3983 | 85.87% | 0.3983 | 86.49% |
| Fine-tune | 8 | 0.3764 | 86.42% | 0.3650 | 86.73% |
| Fine-tune | 9 | 0.3577 | 87.05% | 0.3944 | 85.70% |
| Fine-tune | 10 | 0.3425 | 87.52% | 0.3776 | 86.42% |
| Fine-tune | 11 | 0.3286 | 87.95% | 0.3778 | 86.49% |
| Fine-tune | 12 | 0.3171 | 88.46% | 0.3457 | 87.69% |
| Fine-tune | 13 | 0.3038 | 88.88% | 0.3547 | 86.88% |
| Fine-tune | 14 | 0.2960 | 89.19% | 0.3290 | 87.84% |
| Fine-tune | 15 | 0.2859 | 89.34% | 0.3493 | 87.30% |
| Fine-tune | 16 | 0.2784 | 89.65% | 0.3548 | 87.87% |
| Fine-tune | 17 | 0.2692 | 90.06% | 0.3488 | 87.91% |
| Fine-tune | 18 | 0.2642 | 90.22% | **0.3245** | 88.26% |
| Fine-tune | 19 | 0.2537 | 90.42% | 0.3389 | 88.36% |
| Fine-tune | 20 | 0.2477 | 90.67% | 0.3454 | 88.03% |
| Fine-tune | 21 | 0.2417 | 90.91% | 0.3693 | 87.71% |
| Fine-tune | 22 | 0.2361 | 91.10% | 0.3349 | **88.87%** |
| Fine-tune | 23 | 0.2285 | 91.46% | 0.3339 | 88.08% |

Other logged measurements: **12,636,668 FLOPs/image**, **6,318,334 estimated MACs/image**, and median CPU latency **3.3112 ms/image** at batch size 1. The pasted log ended at "Final evaluation on the test set"; no final test metrics for this run were included. Do not infer them from earlier runs.

## Validation-based comparison

| Stage 2 configuration | Lowest validation loss | Accuracy at that checkpoint | Peak validation accuracy |
|---|---:|---:|---:|
| 20 epochs, batch 32, 32x32 | 0.3551 | 87.75% | 87.75% |
| 30 epochs, batch 32, 32x32 | 0.3456 | 87.70% | 88.01% |
| 30-epoch limit, batch 8, 32x32 | 0.3245 | 88.26% | 88.87% |
| 30-epoch limit, batch 8, 64x64, 30% fine-tuning | **0.2829** | **90.74%** | **91.75%** |
| 30-epoch limit, batch 8, 64x64, 20% fine-tuning, dropout 0.2 | 0.3071 | 89.81% | 90.02% |
| 30-epoch limit, batch 8, 64x64, 20% fine-tuning, dropout 0.3 | 0.2976 | 89.70% | 90.54% |
| Repeat Run 7: batch 8, 64x64, 20% fine-tuning, dropout 0.3 | 0.2948 | 90.03% | 90.52% |

Run 4 at 64x64 produced the best recorded validation loss and accuracy so far, at a higher computation and latency cost. Its training accuracy rose well above validation accuracy and validation loss fluctuated after epoch 8, which is a stronger training/validation gap than in the earlier runs. Use the saved lowest-validation-loss checkpoint for comparisons. Do not use test metrics to select resolution, fine-tuning fraction, epochs, batch size, or other settings.

## Run 4: 64x64 resolution, batch size 8

The model summary confirms 64x64 input and a 2x2x1280 backbone feature map. It used the same 30% fine-tuning fraction, seed, batch size, and learning rates as Run 3. Stage 1 stopped at epoch 5 after its best validation loss at epoch 2; Stage 2 stopped at epoch 13 after its best validation loss at epoch 8. The best Stage 2 validation loss was **0.2829** (epoch 8); the highest validation accuracy was **91.75%** (epoch 13). The Stage 2 checkpoint is selected by validation loss, so it corresponds to epoch 8.

| Stage | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| Head | 1 | 0.5890 | 80.41% | 0.4630 | 84.63% |
| Head | 2 | 0.4957 | 83.45% | **0.4580** | 84.95% |
| Head | 3 | 0.4864 | 83.93% | 0.4963 | 83.61% |
| Head | 4 | 0.4815 | 84.17% | 0.4622 | 84.92% |
| Head | 5 | 0.4755 | 84.35% | 0.4896 | 84.33% |
| Fine-tune | 1 | 0.4080 | 85.76% | 0.3834 | 86.74% |
| Fine-tune | 2 | 0.3191 | 88.49% | 0.3231 | 88.65% |
| Fine-tune | 3 | 0.2689 | 90.25% | 0.3038 | 89.56% |
| Fine-tune | 4 | 0.2312 | 91.36% | 0.3474 | 88.30% |
| Fine-tune | 5 | 0.2022 | 92.60% | 0.2900 | 90.04% |
| Fine-tune | 6 | 0.1795 | 93.32% | 0.2935 | 90.08% |
| Fine-tune | 7 | 0.1588 | 94.03% | 0.3037 | 90.15% |
| Fine-tune | 8 | 0.1389 | 94.82% | **0.2829** | 90.74% |
| Fine-tune | 9 | 0.1238 | 95.34% | 0.3245 | 90.29% |
| Fine-tune | 10 | 0.1112 | 95.84% | 0.2996 | 90.75% |
| Fine-tune | 11 | 0.0986 | 96.28% | 0.3312 | 90.61% |
| Fine-tune | 12 | 0.0881 | 96.72% | 0.3296 | 91.30% |
| Fine-tune | 13 | 0.0797 | 97.02% | 0.3096 | **91.75%** |

The completed log confirms final test evaluation and saved plots under `/content/drive/MyDrive/EN3150_A03/mobilenetv2_64`. These results complete Run 4; they are final reporting metrics, not evidence for further hyperparameter selection.

| Final reported metric | Value |
|---|---:|
| Test loss | 0.2731087803840637 |
| Test accuracy | 0.9103809523809524 (91.04%) |
| Macro precision | 0.9125760558450068 |
| Macro recall | 0.9103809523809524 |
| Macro F1 | 0.9108275607363477 |
| Total parameters | 2,270,794 |
| Trainable parameters | 1,736,650 |
| Reported serialized model size (MB) | 22.485783576965332 |
| Estimated FLOPs/image | 50,162,684 |
| Estimated MACs/image | 25,081,342 |
| Median CPU latency, batch size 1 (ms/image) | 5.529271000000335 |

The saved plots include the final test confusion matrix and per-class scores, both stages' learning curves and epoch timings, and the combined training curve. The completed log has the same epoch values and timing measurements as the earlier partial Run 4 log, so this is the completion of that experiment rather than a new run.

Compared with Run 3 (32x32, batch 8, same fine-tuning fraction), Run 4 improved best validation loss from 0.3245 to 0.2829 and peak validation accuracy from 88.87% to 91.75%. It required about 4x the estimated FLOPs and measured about 1.67x the CPU latency. This is an exploratory resolution comparison; the 32x32 run remains the assignment-comparable baseline unless all compared models are evaluated at 64x64.

## Run 5: 64x64, batch size 8, 20% fine-tuning

This run changes the fine-tuning fraction from 0.3 to 0.2. It selects the last 30 of 154 backbone layers; 19 remain trainable after freezing BatchNormalization. The local script at review time still monitors validation loss for both checkpointing and early stopping, uses dropout 0.2, and retains the learning rates from Run 4. This is a fine-tuning-fraction experiment; it does not test validation-accuracy checkpoint selection.

Stage 1 repeats Run 4's epoch metrics and stops after epoch 5. Stage 2 stops after epoch 11: epoch 6 has the best validation loss, followed by five epochs without a lower loss. The selected checkpoint has validation loss **0.3071** and validation accuracy **89.81%**. Peak validation accuracy is **90.02%** at epoch 11.

| Stage | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Logged epoch time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Head | 1 | 0.5890 | 80.41% | 0.4630 | 84.63% | 59 |
| Head | 2 | 0.4957 | 83.45% | 0.4580 | 84.95% | 43 |
| Head | 3 | 0.4864 | 83.93% | 0.4963 | 83.61% | 43 |
| Head | 4 | 0.4815 | 84.17% | 0.4622 | 84.92% | 44 |
| Head | 5 | 0.4755 | 84.35% | 0.4896 | 84.33% | 45 |
| Fine-tune | 1 | 0.3993 | 85.94% | 0.3668 | 87.45% | 65 |
| Fine-tune | 2 | 0.3121 | 88.66% | 0.3299 | 88.37% | 47 |
| Fine-tune | 3 | 0.2623 | 90.42% | 0.3252 | 88.90% | 47 |
| Fine-tune | 4 | 0.2226 | 91.68% | 0.3363 | 88.71% | 45 |
| Fine-tune | 5 | 0.1936 | 92.85% | 0.3107 | 89.41% | 46 |
| Fine-tune | 6 | 0.1695 | 93.57% | **0.3071** | 89.81% | 46 |
| Fine-tune | 7 | 0.1467 | 94.47% | 0.3476 | 89.38% | 46 |
| Fine-tune | 8 | 0.1296 | 95.21% | 0.3149 | 90.00% | 45 |
| Fine-tune | 9 | 0.1130 | 95.84% | 0.3377 | 89.81% | 46 |
| Fine-tune | 10 | 0.1010 | 96.23% | 0.3458 | 89.81% | 82 |
| Fine-tune | 11 | 0.0883 | 96.73% | 0.3429 | **90.02%** | 45 |

The completed log saves outputs to `/content/drive/MyDrive/EN3150_A03/mobilenetv2_64_0.2_unfreeze`.

| Final reported metric | Value |
|---|---:|
| Test loss | 0.29469308257102966 |
| Test accuracy | 0.9019047619047619 (90.19%) |
| Macro precision | 0.9033049799298618 |
| Macro recall | 0.9019047619047619 |
| Macro F1 | 0.9020848266802673 |
| Total parameters | 2,270,794 |
| Trainable parameters | 1,523,530 |
| Reported serialized model size (MB) | 20.856597900390625 |
| Estimated FLOPs/image | 50,162,684 |
| Estimated MACs/image | 25,081,342 |
| Median CPU latency, batch size 1 (ms/image) | 5.392482499928519 |

Validation-based conclusion: Run 4 at 30% remains preferable to this 20% run under the current selection rule. It has lower validation loss (0.2829 versus 0.3071) and higher validation accuracy at its selected checkpoint (90.74% versus 89.81%). This is evidence from these runs, not proof that 30% is universally optimal. Test metrics above are retained for reporting only. Freezing additional layers reduces trainable parameters but does not change the forward inference graph, hence the unchanged FLOPs/MACs. Small differences in CPU timing should not be attributed to the freezing fraction from one measurement.

## Run 6: 64x64, batch size 8, 20% fine-tuning, dropout 0.3

This run changes dropout from 0.2 (Run 5) to 0.3, retaining the 20% fine-tuning fraction. It selects 30 backbone layers, with 19 trainable after freezing BatchNormalization. Checkpointing and early stopping still monitor validation loss. Stage 1 stops after epoch 5 and restores its epoch 2 weights. Stage 2 stops after epoch 10: epoch 5 has the lowest validation loss, and epochs 6-10 do not improve it. The restored checkpoint has validation loss **0.2976** and validation accuracy **89.70%**. Peak validation accuracy is **90.54%** at epoch 10; that is not the restored lowest-loss checkpoint.

| Stage | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Logged epoch time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Head | 1 | 0.6273 | 79.19% | 0.4712 | 84.35% | 60 |
| Head | 2 | 0.5400 | 82.25% | **0.4577** | 84.74% | 44 |
| Head | 3 | 0.5305 | 82.78% | 0.4836 | 83.70% | 82 |
| Head | 4 | 0.5312 | 82.91% | 0.4606 | 84.73% | 44 |
| Head | 5 | 0.5246 | 83.08% | 0.4784 | 84.67% | 81 |
| Fine-tune | 1 | 0.4277 | 84.93% | 0.3688 | 87.30% | 64 |
| Fine-tune | 2 | 0.3339 | 87.91% | 0.3302 | 87.98% | 51 |
| Fine-tune | 3 | 0.2827 | 89.56% | 0.3124 | 89.10% | 49 |
| Fine-tune | 4 | 0.2451 | 90.89% | 0.3240 | 89.01% | 47 |
| Fine-tune | 5 | 0.2173 | 91.99% | **0.2976** | 89.70% | 47 |
| Fine-tune | 6 | 0.1928 | 92.79% | 0.3013 | 89.88% | 48 |
| Fine-tune | 7 | 0.1715 | 93.57% | 0.3075 | 89.93% | 46 |
| Fine-tune | 8 | 0.1519 | 94.34% | 0.2980 | 89.96% | 82 |
| Fine-tune | 9 | 0.1367 | 94.93% | 0.3396 | 89.82% | 47 |
| Fine-tune | 10 | 0.1240 | 95.24% | 0.3121 | **90.54%** | 46 |

Outputs were saved to `/content/drive/MyDrive/EN3150_A03/mobilenetv2_64_0.2_unfreeze_0.3_dropout`.

| Final reported metric | Value |
|---|---:|
| Test loss | 0.29604989290237427 |
| Test accuracy | 0.9028571428571428 (90.29%) |
| Macro precision | 0.9030557901663487 |
| Macro recall | 0.9028571428571428 |
| Macro F1 | 0.9023445005437528 |
| Total parameters | 2,270,794 |
| Trainable parameters | 1,523,530 |
| Reported serialized model size (MB) | 20.856597900390625 |
| Estimated FLOPs/image | 50,162,684 |
| Estimated MACs/image | 25,081,342 |
| Median CPU latency, batch size 1 (ms/image) | 5.633259000205726 |

### Run 6 test results by class

The subsequently supplied per-class report matches Run 6's macro scores. Exact values are preserved in `documents/experiment_data/run06_test_classification_report.json`. The following table ranks classes by F1 score. These test results describe the evaluated model; use validation class scores and a validation confusion matrix for further tuning.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Trouser | 98.84% | 97.62% | 98.23% |
| Bag | 97.91% | 98.29% | 98.10% |
| Sandal | 96.78% | 97.43% | 97.10% |
| Ankle boot | 96.74% | 96.00% | 96.37% |
| Sneaker | 95.99% | 95.71% | 95.85% |
| Dress | 87.70% | 92.38% | 89.98% |
| Pullover | 85.85% | 86.10% | 85.97% |
| T-shirt/top | 88.35% | 80.86% | 84.44% |
| Coat | 79.56% | 88.95% | 83.99% |
| Shirt | 75.34% | 69.52% | 72.31% |

Trouser and Bag have the highest F1 scores. Shirt has the lowest F1 and recall: about 30.48% of actual Shirt images were assigned another label. Coat precision is 79.56%, so about 20.44% of predictions labeled Coat belong to another class. These scores alone do not identify the actual confusion pairs.

The supplied confusion-matrix JSON contains nine objects and eight visible keys per object, with numeric column names rather than a complete ten-class matrix. It cannot be reliably reconstructed as the original 10x10 matrix. Obtain the original CSV with all ten rows and ten columns to describe exact test confusion pairs; use a separate validation matrix for decisions about model changes.

## Run 7: repeated training while attempting validation analysis

The supplied output identifies `MobileNetV2/model.py`, runs training epochs in both stages, and ends with final test metrics. It is a completed training run, not the standalone `analyze_validation.py` analysis. Renaming the output folder to include `analise` does not change which dataset is evaluated. No new validation-classification analysis report is shown. The complete raw log, including every epoch's scores and timings, is preserved in `documents/experiment_data/run07_training_log.txt`.

The log confirms input size 64, batch size 8 (6,125 batches for 49,000 images), fine-tuning fraction 0.2, and 19 trainable backbone layers after freezing BatchNormalization. The output-folder name identifies dropout 0.3, matching the current script. No further hyperparameter change has been established; treat this as a repeat rather than attributing its different results to the analysis script.

Stage 2 ends at epoch 11. The lowest validation loss is **0.2948** at epoch 6, with validation accuracy **90.03%**. The following five epochs have higher validation loss. Peak validation accuracy is **90.52%** at epoch 11. Existing Run 4 still has lower validation loss (0.2829) and higher accuracy at its selected checkpoint (90.74%). Test scores below are for reporting, not selection.

| Final reported metric | Value |
|---|---:|
| Test loss | 0.28698641061782837 |
| Test accuracy | 0.9048571428571428 (90.49%) |
| Macro precision | 0.9047446419440666 |
| Macro recall | 0.904857142857143 |
| Macro F1 | 0.9045991925095624 |
| Total parameters | 2,270,794 |
| Trainable parameters | 1,523,530 |
| Reported serialized model size (MB) | 20.856597900390625 |
| Estimated FLOPs/image | 50,162,684 |
| Estimated MACs/image | 25,081,342 |
| Median CPU latency, batch size 1 (ms/image) | 5.003966999993281 |

Outputs were saved to `/content/drive/MyDrive/EN3150_A03/mobilenetv2_64_0.2_unfreeze_0.3_dropout_analise`. At the time of Run 7, the intended validation-only analysis used the optional `MobileNetV2/analyze_validation.py` script with the saved Run 4 checkpoint. That helper has since been removed: the current `MobileNetV2/model.py` saves validation metrics, class reports, and confusion matrices in `stage1_validation` and `stage2_validation` during training. Final test evaluation remains a separate `--test_with_test_data "RUN_DIR"` command in the same model file.

Validation-based conclusion: dropout 0.3 gives mixed results compared with Run 5. Lowest validation loss improves from 0.3071 to 0.2976 and peak validation accuracy rises from 90.02% to 90.54%, but the accuracy of the checkpoint selected by validation loss declines slightly from 89.81% to 89.70%. Early stopping at epoch 10 is not itself a sign of a worse run; it reflects five consecutive epochs without a lower validation loss. Among Runs 1-7, Run 4 (64x64, 30% fine-tuning under the old selection rule, dropout 0.2) has the best recorded validation loss and selected-checkpoint accuracy. The final test values are recorded for reporting only and are not used for this conclusion.

## Run 8: corrected non-BN quota and validation-only checkpoint selection

Run directory: `/content/drive/MyDrive/EN3150_A03/mobilenetv2_64/run_20261018_171840Z_um1rjogw`.

This is the first supplied completed run using the corrected selection rule: target `int(154 * 0.3) = 46` layers and fill that quota with non-BatchNormalization layers from the end of the backbone. The log confirms **46 marked-trainable non-BN backbone layers, of which 24 have trainable weights**. The earlier 30% window rule left only 30 layers marked trainable after BatchNormalization was frozen. Treat this as a new configuration; the improvement cannot be attributed to the quota change alone because best-head restoration and the training workflow also changed.

The log confirms 64x64 input, batch size 8 (6,125 batches for the shared 49,000 training examples), and fine-tuning fraction 0.3. Local model.py defaults are dropout 0.2, seed 42, alpha 1.0, head/fine-tuning learning rates 1e-3/1e-5 and patience 3/5; these settings are not all printed in the supplied log, so confirm them from this run's saved config.json before final reporting.

**Stage 1:** 5 of 10 possible epochs completed. Best validation loss **0.4580** was at epoch **2**, with displayed validation accuracy **84.95%**. Three later epochs did not improve the loss, consistent with head patience 3; best head weights were restored before Stage 2.

**Stage 2:** 10 of 30 possible epochs completed. Best validation loss **0.23984792828559875** was at epoch **5**, with selected-checkpoint validation accuracy **91.25714285714286%**. Epochs 6-10 did not beat that loss, consistent with fine-tuning patience 5. Peak displayed validation accuracy was **92.11% at epoch 10**, but that epoch's validation loss was 0.2507, so the epoch-5 checkpoint was restored and selected. Report 91.26% as selected-model validation accuracy, not 92.11%.

| Stage | Epoch | Displayed train loss | Displayed train accuracy | Validation loss | Validation accuracy | Displayed seconds |
|---|---:|---:|---:|---:|---:|---:|
| Head | 1 | 0.5890 | 80.41% | 0.4630 | 84.63% | 57 |
| Head | 2 | 0.4957 | 83.45% | 0.4580 | 84.95% | 41 |
| Head | 3 | 0.4864 | 83.93% | 0.4963 | 83.61% | 40 |
| Head | 4 | 0.4815 | 84.17% | 0.4622 | 84.92% | 40 |
| Head | 5 | 0.4755 | 84.35% | 0.4896 | 84.33% | 40 |
| Fine-tune | 1 | 0.3973 | 85.99% | 0.3382 | 88.06% | 67 |
| Fine-tune | 2 | 0.2999 | 89.08% | 0.2923 | 89.37% | 46 |
| Fine-tune | 3 | 0.2470 | 91.03% | 0.2619 | 90.64% | 46 |
| Fine-tune | 4 | 0.2105 | 92.21% | 0.2725 | 90.03% | 45 |
| Fine-tune | 5 | 0.1837 | 93.26% | 0.2398 | 91.26% | 46 |
| Fine-tune | 6 | 0.1615 | 93.93% | 0.2500 | 91.26% | 45 |
| Fine-tune | 7 | 0.1433 | 94.73% | 0.2659 | 91.22% | 45 |
| Fine-tune | 8 | 0.1239 | 95.29% | 0.2516 | 91.55% | 81 |
| Fine-tune | 9 | 0.1106 | 95.85% | 0.2911 | 91.68% | 45 |
| Fine-tune | 10 | 0.0990 | 96.32% | 0.2507 | 92.11% | 49 |

| Selected-model metric | Exact logged value |
|---|---:|
| `split` | validation |
| `images` | 10500 |
| `validation_loss` | 0.23984792828559875 |
| `validation_accuracy` | 0.9125714285714286 |
| `macro_precision` | 0.9130564106193834 |
| `macro_recall` | 0.9125714285714286 |
| `macro_f1` | 0.9127210513890006 |
| `weighted_f1` | 0.9127210513890007 |
| `total_parameters` | 2270794 |
| `trainable_parameters` | 2000650 |
| `non_trainable_parameters` | 270144 |
| `training_checkpoint_size_mib` | 24.506260871887207 |
| `inference_model_size_mib` | 9.215943336486816 |
| `weight_data_size_mib` | 8.662391662597656 |
| `flops_per_image` | 50162684 |
| `mflops_per_image` | 50.162684 |
| `estimated_macs_per_image` | 25081342 |
| `cpu_inference_median_ms_per_image_batch1` | 4.544919999943886 |
| `cpu_latency_warmup_runs` | 10 |
| `cpu_latency_measured_runs` | 100 |
| `selected_stage` | fine_tuning |
| `input_size` | 64 |
| `fine_tune_fraction` | 0.3 |

The inference file is **9.21594 MiB**, the training checkpoint is **24.50626 MiB**, and raw weight data occupies **8.66239 MiB**. These are different measurements: do not compare this inference-file size directly with the older optimizer-containing checkpoint sizes as evidence of an architectural size reduction. The total parameter count and FLOPs remain unchanged from the earlier 64x64 runs. CPU latency is the median single-image synthetic-input measurement on the Colab CPU; the reported value is **4.54492 ms/image**, with 10 warmup and 100 measured runs.

Compared with Run 4's rounded best validation loss 0.2829 and selected-checkpoint accuracy 90.74%, this run has lower validation loss (0.23985) and about **0.52 percentage points higher selected-checkpoint accuracy** (91.26%). It is the best supplied MobileNet run by the established validation-loss selection criterion. One run does not establish repeatability or isolate which code change caused the gain.

Later epochs reduce training loss while validation loss fluctuates above the epoch-5 minimum. This suggests increasing overfitting after the selected checkpoint, although validation accuracy continues to improve; model confidence and accuracy are different quantities. Early stopping preserves the lower-loss checkpoint.

**No test results exist in this supplied log.** Its final metrics explicitly say `split: validation`, and the process prints that test data has not been evaluated. If this configuration is selected after validation comparison, final testing is:

```python
!python MobileNetV2/model.py --test_with_test_data "/content/drive/MyDrive/EN3150_A03/mobilenetv2_64/run_20261018_171840Z_um1rjogw"
```

The process completed despite GPU initialization/timing and profiler-deprecation messages. The pretrained-weight message says the 224-resolution ImageNet weight file was loaded; the actual model summary confirms its input remains 64x64 and its final backbone feature map is 2x2x1280.

The complete supplied log is preserved byte-for-byte in `documents/experiment_data/run08_training_log.txt`. Exact selected metrics, every displayed epoch and provenance notes are saved in `documents/experiment_data/run08_validation_summary.json`. Validation per-class JSON and matrix contents were not included in this log; their saved locations are printed under stage1_validation and stage2_validation.
