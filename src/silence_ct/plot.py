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

# validation AUC lines (one per region type)
AUC_LINES = [
    ("val_auc_heart", "heart", "#6a4ce2", "-"),
    ("val_auc_esophagus", "esophagus", "#00b894", "-"),
    ("val_auc_mean", "mean (all regions)", TEXT_COLOR, "--"),
]

STYLE = {
    "font.size": 10,
    "axes.edgecolor": MUTED_COLOR,
    "axes.labelcolor": MUTED_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
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

      - the best epoch (saved model) is a dot on the validation line with its value

    Args:
        ax: matplotlib axis
        epochs: epoch numbers
        train_values: train metric per epoch
        val_values: validation metric per epoch
        best_epoch: epoch saved by train.py (highest mean validation AUC)
        title: panel title
        ylabel: y-axis label
    """
    ax.plot(epochs, train_values, color=TRAIN_COLOR, label="train")
    ax.plot(epochs, val_values, color=VAL_COLOR, label="validation")

    # best epoch: dot with a white ring, value written beside it
    best_value = val_values[epochs == best_epoch].iloc[0]
    ax.scatter([best_epoch], [best_value], s=60, color=VAL_COLOR,
               edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"{best_value:.3f}", (best_epoch, best_value),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=9, color=TEXT_COLOR)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.margins(x=0.02)


def draw_auc(ax, log, best_epoch):
    """
    Draw validation ROC-AUC per epoch for heart, esophagus and the mean of all regions.

      - the best epoch (highest mean AUC, the saved model) is a dot on the mean line

    Args:
        ax: matplotlib axis
        log: training log table
        best_epoch: epoch saved by train.py
    """
    for column, label, color, style in AUC_LINES:
        ax.plot(log["epoch"], log[column], color=color, linestyle=style, label=label)

    best_value = log.loc[log["epoch"] == best_epoch, "val_auc_mean"].iloc[0]
    ax.scatter([best_epoch], [best_value], s=60, color=TEXT_COLOR,
               edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"best {best_value:.3f}", (best_epoch, best_value),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=9, color=TEXT_COLOR)

    ax.set_title("Validation ROC-AUC")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ROC-AUC (0.5 = guessing)")
    ax.legend(loc="best")
    ax.margins(x=0.02)


def plot_training(run_dir):
    """
    Draw training curves for one run and save them as training_curves.png.

    Three panels:
      - state loss: train and validation, lower is better
      - validation ROC-AUC: heart, esophagus and mean, higher is better
      - cosine: train and validation, higher is better (1 = perfect)
    Train numbers use dropout, so they are not exactly comparable to validation.
    Needs a log with AUC columns (runs trained after the AUC change).

    Args:
        run_dir: run folder with log.csv

    Returns:
        path of the saved figure
    """
    log = pd.read_csv(Path(run_dir) / "log.csv")
    epochs = log["epoch"]
    best_epoch = int(log.loc[log["val_auc_mean"].idxmax(), "epoch"])

    with plt.rc_context(STYLE):
        fig, (loss_ax, auc_ax, cosine_ax) = plt.subplots(1, 3, figsize=(15, 3.8))

        draw_panel(loss_ax, epochs, log["train_state_loss"], log["val_state_loss"], best_epoch,
                   "State loss", "cross-entropy")
        draw_auc(auc_ax, log, best_epoch)
        draw_panel(cosine_ax, epochs, 1 - log["train_vector_loss"], log["val_cosine"], best_epoch,
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