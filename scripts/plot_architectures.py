"""Architecture diagrams for Model A and Model B (report figures).

Feature maps are drawn as 3D blocks: height = spatial size (28/14/7), thickness = channels.
Writes results/architecture/model_a_architecture.png and model_b_architecture.png.

    python scripts/plot_architectures.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "architecture"

STD = "#9cb3d4"      # standard convolution output
SEP = "#7cc4a4"      # depthwise separable convolution output
POOL = "#d9d9d9"     # pooling output
INPUT = "#f2f2f2"
DENSE = "#f2c48d"
EDGE = "#3a3a3a"

LABEL_Y = -1.72     # operation labels sit below the tallest (28x28) block
BRACKET_Y = -2.95


def block(ax, x, size, channels, color):
    """Draw a feature map as a 3D block; returns its right edge x."""
    h = 0.11 * size                          # 28 -> 3.08, 14 -> 1.54, 7 -> 0.77
    t = 0.12 + 0.11 * math.log2(channels)    # thickness grows slowly with channels
    d = 0.35 * h                             # oblique depth (the other spatial axis)
    y = -h / 2
    front = Rectangle((x, y), t, h, facecolor=color, edgecolor=EDGE, linewidth=1)
    top = Polygon([(x, y + h), (x + d, y + h + d * 0.6), (x + t + d, y + h + d * 0.6), (x + t, y + h)],
                  closed=True, facecolor=color, edgecolor=EDGE, linewidth=1, alpha=0.85)
    side = Polygon([(x + t, y), (x + t + d, y + d * 0.6), (x + t + d, y + h + d * 0.6), (x + t, y + h)],
                   closed=True, facecolor=color, edgecolor=EDGE, linewidth=1, alpha=0.7)
    for p in (side, top, front):
        ax.add_patch(p)
    return x + t + d, y + h + d * 0.6


def vector(ax, x, units, color, max_units=128):
    """Draw a 1-D layer (GAP / dense) as a thin bar; returns its right edge x and top y."""
    h = 0.5 + 2.6 * units / max_units
    ax.add_patch(Rectangle((x, -h / 2), 0.22, h, facecolor=color, edgecolor=EDGE, linewidth=1))
    return x + 0.22, h / 2


def arrow(ax, x0, x1):
    ax.add_patch(FancyArrowPatch((x0 + 0.08, 0), (x1 - 0.08, 0), arrowstyle="-|>", mutation_scale=12,
                                 color=EDGE, linewidth=1))


def draw(model, title, subtitle, conv2, conv3, path):
    fig, ax = plt.subplots(figsize=(18, 6.6))
    ax.set_ylim(-3.45, 2.9)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.895, subtitle, ha="center", fontsize=11, color="#444444")

    # (label above block, size, channels, color, op label below the arrow INTO this block)
    stages = [
        ("Input\n28×28×1", 28, 1, INPUT, None),
        ("28×28×32", 28, 32, STD, "Conv 3×3, 32\nReLU\n320 params"),
        ("14×14×32", 14, 32, POOL, "MaxPool\n2×2"),
        ("14×14×64", 14, 64, conv2[0], conv2[1]),
        ("7×7×64", 7, 64, POOL, "MaxPool\n2×2"),
        ("7×7×128", 7, 128, conv3[0], conv3[1]),
    ]
    x, prev = 0.0, None
    for label, size, ch, color, op in stages:
        if prev is not None:
            x0 = prev
            x += 2.3
            arrow(ax, x0, x)
            ax.text((x0 + x) / 2, LABEL_Y, op, ha="center", va="top", fontsize=9.5, linespacing=1.3)
        right, top = block(ax, x, size, ch, color)
        ax.text((x + right) / 2, top + 0.15, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
        prev, x = right, right

    feat_end = x
    heads = [("128", 128, POOL, "Global avg\npool\n0 params"),
             ("64", 64, DENSE, "Dense 64\nReLU\n8,256 params"),
             ("10", 10, DENSE, "Dense 10\nsoftmax\n650 params")]
    for label, units, color, op in heads:
        x0 = prev
        x += 1.75
        arrow(ax, x0, x)
        ax.text((x0 + x) / 2, LABEL_Y, op, ha="center", va="top", fontsize=9.5, linespacing=1.3)
        right, top = vector(ax, x, units, color)
        ax.text((x + right) / 2, top + 0.15, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
        prev, x = right, right

    # class outputs
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    arrow(ax, prev, prev + 0.6)
    for i, name in enumerate(classes):
        ax.text(prev + 0.7, 0.9 - i * 0.2, name, fontsize=8.5, va="center")

    end = prev + 2.4
    ax.set_xlim(-0.4, end + 0.3)

    # bracket labels like the brief's figure
    for x0, x1, name in [(0, feat_end, "FEATURE LEARNING"), (feat_end + 0.4, end, "CLASSIFICATION")]:
        ax.plot([x0, x1], [BRACKET_Y, BRACKET_Y], color="#777777", linewidth=1)
        ax.text((x0 + x1) / 2, BRACKET_Y - 0.12, name, ha="center", va="top", fontsize=10, color="#555555", fontweight="bold")

    # legend
    legend = [(INPUT, "Input image"), (STD, "Standard conv output"), (POOL, "Pooling output")]
    if model == "B":
        legend.insert(2, (SEP, "Depthwise separable conv output"))
    legend.append((DENSE, "Dense layer"))
    handles = [Patch(facecolor=c, edgecolor=EDGE, label=t) for c, t in legend]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=len(handles),
              frameon=False, fontsize=9.5, handlelength=1.4, columnspacing=2.2)

    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw("A", "Model A: standard CNN",
         "101,578 trainable parameters  ·  7.46M MACs per image  ·  all 3×3 convolutions are standard Conv2D",
         (STD, "Conv 3×3, 64\nReLU\n18,496 params"),
         (STD, "Conv 3×3, 128\nReLU\n73,856 params"),
         OUT / "model_a_architecture.png")
    draw("B", "Model B: lightweight CNN with depthwise separable convolutions",
         "20,522 trainable parameters  ·  1.12M MACs per image  ·  layers 2 and 3 are depthwise 3×3 + pointwise 1×1",
         (SEP, "Separable conv\nDW 3×3 + PW 1×1\n64, ReLU\n2,400 params"),
         (SEP, "Separable conv\nDW 3×3 + PW 1×1\n128, ReLU\n8,896 params"),
         OUT / "model_b_architecture.png")


if __name__ == "__main__":
    main()
