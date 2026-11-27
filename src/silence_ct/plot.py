"""
Training curves

Plots loss and cosine similarity per epoch for train and validation, from the
log.csv that train.py writes in each run folder, and saves the figure there.

Run: PYTHONPATH=src python -m silence_ct.plot --run_dir <path to run folder>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# colours checked for colour-blind readers; text stays grey, not the line colour
TRAIN_COLOR = "#257de0" 
VAL_COLOR = "#f2a600"
TEXT_COLOR = "#191919"
MUTED_COLOR = "#666666"
GRID_COLOR = "#e6e6e6"

STYLE = {
    "font.size": 10,
    "axes.edgecolor": MUTED_COLOR,
    "axes.labelcolor": MUTED_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "legend.frameon": False,
    "lines.linewidth": 2,
}


def draw_panel(ax, epochs, train_values, val_values, best_epoch, title, ylabel):
    """
    Draw one panel with a train line, a validation line and the best epoch.

      - each line is named at its last point, so the reader does not need the legend
      - the best epoch is a dot on the validation line with its value next to it

    Args:
        ax: matplotlib axis
        epochs: epoch numbers
        train_values: train metric per epoch
        val_values: validation metric per epoch
        best_epoch: epoch saved by train.py (lowest validation loss)
        title: panel title
        ylabel: y-axis label
    """
    ax.plot(epochs, train_values, color=TRAIN_COLOR, label="train")
    ax.plot(epochs, val_values, color=VAL_COLOR, label="validation")

    # names at the end of each line
    last_epoch = epochs.iloc[-1]
    ax.annotate("train", (last_epoch, train_values.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=MUTED_COLOR)
    ax.annotate("validation", (last_epoch, val_values.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=MUTED_COLOR)

    # best epoch: dot with a white ring, value written beside it
    best_value = val_values[epochs == best_epoch].iloc[0]
    ax.scatter([best_epoch], [best_value], s=60, color=VAL_COLOR,
               edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"best {best_value:.3f}\n(epoch {best_epoch})", (best_epoch, best_value),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=9, color=TEXT_COLOR)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.margins(x=0.08)


def plot_training(run_dir):
    """
    Draw training curves for one run and save them as training_curves.png.

    Two panels:
      - loss:   train and validation (1 - cosine), lower is better
      - cosine: train and validation, higher is better (1 = perfect)
    Train numbers use dropout, so they are not exactly comparable to validation.

    Args:
        run_dir: run folder with log.csv, e.g. silence_ct/runs/regression_both

    Returns:
        path of the saved figure
    """
    log = pd.read_csv(Path(run_dir) / "log.csv")
    epochs = log["epoch"]
    best_epoch = int(log.loc[log["val_loss"].idxmin(), "epoch"])

    with plt.rc_context(STYLE):
        fig, (loss_ax, cosine_ax) = plt.subplots(1, 2, figsize=(10, 3.8))

        draw_panel(loss_ax, epochs, log["train_loss"], log["val_loss"], best_epoch,
                   "Loss", "1 − cosine")
        draw_panel(cosine_ax, epochs, 1 - log["train_loss"], log["val_cosine"], best_epoch,
                   "Cosine similarity", "cosine")

        fig.suptitle(Path(run_dir).name, x=0.01, ha="left", color=MUTED_COLOR, fontsize=9)
        fig.tight_layout()

        out_path = Path(run_dir) / "training_curves.png"
        fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)
    return out_path


if __name__ == "__main__":
    # PYTHONPATH=src python -m silence_ct.plot --run_dir data/runs/regression_both
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    print("saved:", plot_training(args.run_dir))