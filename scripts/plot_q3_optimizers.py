"""Q3 figures for Model B: chosen optimizer vs SGD / SGD+momentum, momentum sweep, learning-rate sweep.

Reads the per-run files in results/model_b/all_runs/ (from the Colab T4 sweep) and writes
results/model_b/q3_figures/. All comparisons use validation data only.

    python scripts/plot_q3_optimizers.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "model_b" / "all_runs"
OUT = ROOT / "results" / "model_b" / "q3_figures"

# Chosen configuration (best final validation accuracy at 20 epochs, batch 128) and its baselines.
OPTIMIZER_RUNS = {
    "RMSprop, lr 0.003 (chosen)": "model_b_rmsprop_lr0.003_bs128_e20",
    "Adam, lr 0.001": "model_b_adam_lr0.001_bs128_e20",
    "SGD + momentum 0.9, lr 0.01": "model_b_sgd_momentum_lr0.01_bs128_e20_m0.9",
    "SGD, lr 0.01": "model_b_sgd_lr0.01_bs128_e20",
}
MOMENTUM_RUNS = {
    0.0: "model_b_sgd_lr0.01_bs128_e20",
    0.5: "model_b_sgd_momentum_lr0.01_bs128_e20_m0.5",
    0.9: "model_b_sgd_momentum_lr0.01_bs128_e20_m0.9",
    0.99: "model_b_sgd_momentum_lr0.01_bs128_e20_m0.99",
}
CONVERGED = 0.80   # validation accuracy used to measure speed of convergence

# One colour per configuration, used in every figure so the report reads consistently.
COLORS = {
    "model_b_rmsprop_lr0.003_bs128_e20": "tab:blue",
    "model_b_adam_lr0.001_bs128_e20": "tab:orange",
    "model_b_sgd_momentum_lr0.01_bs128_e20_m0.9": "tab:green",
    "model_b_sgd_lr0.01_bs128_e20": "tab:red",
    "model_b_sgd_momentum_lr0.01_bs128_e20_m0.5": "tab:purple",
    "model_b_sgd_momentum_lr0.01_bs128_e20_m0.99": "tab:brown",
    "rmsprop": "tab:blue",
    "adam": "tab:orange",
}


def load(run):
    history = pd.read_csv(RUNS / run / "history.csv")
    metrics = json.loads((RUNS / run / "evaluation.json").read_text())
    return history, metrics


def epochs_to(history, threshold=CONVERGED):
    hit = history.index[history["val_accuracy"] >= threshold]
    return int(history["epoch"][hit[0]]) if len(hit) else None


def plot_curves(runs, title, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    for label, run in runs.items():
        history, _ = load(run)
        ax_loss.plot(history["epoch"], history["val_loss"], label=label, linewidth=2, color=COLORS[run])
        ax_acc.plot(history["epoch"], history["val_accuracy"], label=label, linewidth=2, color=COLORS[run])
    ax_loss.set_ylabel("Validation loss")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.axhline(CONVERGED, color="grey", linestyle=":", linewidth=1)
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(0, 21, 2))
        ax.grid(alpha=0.3)
    # One shared legend under both panels, so it never hides the early epochs where curves differ most.
    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_learning_rates(path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for optimizer in ["adam", "rmsprop"]:
        points = []
        for run_dir in sorted(RUNS.glob(f"model_b_{optimizer}_lr*_bs128_e20")):
            _, metrics = load(run_dir.name)
            points.append((metrics["learning_rate"], metrics["final_val_accuracy"]))
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", linewidth=2,
                color=COLORS[optimizer], label="Adam" if optimizer == "adam" else "RMSprop")
    ax.set_xscale("log")
    ax.set_xticks([0.0003, 0.001, 0.002, 0.003, 0.004], ["0.0003", "0.001", "0.002", "0.003", "0.004"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Final validation accuracy (20 epochs)")
    ax.set_title("Model B: learning-rate sweep, batch 128")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_curves(OPTIMIZER_RUNS, "Model B: chosen optimizer vs SGD and SGD with momentum (batch 128, 20 epochs)",
                OUT / "optimizer_comparison.png")
    plot_curves({f"momentum {m:g}" + (" (plain SGD)" if m == 0 else ""): run for m, run in MOMENTUM_RUNS.items()},
                "Model B: effect of momentum (SGD, lr 0.01, batch 128, 20 epochs)",
                OUT / "momentum_sweep.png")
    plot_learning_rates(OUT / "learning_rate_sweep.png")

    rows = []
    for label, run in {**OPTIMIZER_RUNS, **{f"SGD + momentum {m:g}, lr 0.01": r for m, r in MOMENTUM_RUNS.items() if m in (0.5, 0.99)}}.items():
        history, metrics = load(run)
        rows.append({
            "configuration": label,
            "run": run,
            "final_val_loss": round(float(history["val_loss"].iloc[-1]), 4),
            "final_val_accuracy": round(metrics["final_val_accuracy"], 4),
            f"epochs_to_{int(CONVERGED * 100)}pct_val_accuracy": epochs_to(history),
            "mean_epoch_time_s": metrics["mean_epoch_time_s"],
            "test_accuracy": round(metrics["test_accuracy"], 4),
        })
    summary = pd.DataFrame(rows)
    # Whole epochs; empty where the run never reached the threshold within 20 epochs.
    column = f"epochs_to_{int(CONVERGED * 100)}pct_val_accuracy"
    summary[column] = summary[column].astype("Int64")
    summary.to_csv(OUT / "q3_optimizer_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved figures and summary to {OUT}")


if __name__ == "__main__":
    main()
