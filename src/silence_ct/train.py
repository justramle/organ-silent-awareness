"""
Training

Trains SilenceModel to predict the text vectors of hidden regions. The 5,000 train
scans are split into train (90%) and validation (10%); the test set (1,564 valid
scans) is not used here. The best model on validation is saved.

Run: PYTHONPATH=src python -m silence_ct.train --config configs/config.yaml
Ablation example: ... --geometry none
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from silence_ct.data.dataset import SilenceDataset
from silence_ct.losses import cosine_loss
from silence_ct.models.model import SilenceModel


def load_config():
    """
    Read the YAML config, then apply any command-line overrides.

    Overrides are handy for quick tests and ablations without editing the file:
      - --root data --split valid --epochs 2   -> quick test on the Mac
      - --geometry none                        -> ablation run

    Returns:
        config: dict of settings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--root")
    parser.add_argument("--split")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--geometry")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.root:
        config["data_root"] = args.root
    if args.split:
        config["split"] = args.split
    if args.epochs:
        config["epochs"] = args.epochs
    if args.geometry:
        config["geometry"] = args.geometry
    return config


def split_train_val(dataset, val_fraction, seed):
    """
    Split scan indices into train and validation, stratified by heart and esophagus state.

    The 5,000 train scans are not random (all thyroid-written and esophagus-abnormal
    scans were included), so a stratified split keeps the same share of each group:
      - group = heart state + esophagus state (abnormal / normal / silent each)
      - groups with fewer than 2 scans -> "other" (needed for a stratified split)
      - too few scans to keep every group (local test) -> random split

    Args:
        dataset: SilenceDataset
        val_fraction: share of scans for validation (e.g. 0.1)
        seed: random seed, so the split is the same every run

    Returns:
        train_indices: list of indices
        val_indices: list of indices
    """
    heart = dataset.regions.index("heart")
    esophagus = dataset.regions.index("esophagus")

    groups = []
    for scan in dataset.scans:
        _, state = dataset.text[scan]
        groups.append(f"{state[heart]}_{state[esophagus]}")

    # merge groups that are too small to split
    counts = Counter(groups)
    for i, group in enumerate(groups):
        if counts[group] < 2:
            groups[i] = "other"

    # if "other" is still too small, add it to the biggest group
    counts = Counter(groups)
    if 0 < counts["other"] < 2:
        biggest = counts.most_common(1)[0][0]
        for i, group in enumerate(groups):
            if group == "other":
                groups[i] = biggest

    indices = list(range(len(dataset)))

    # tiny data (local test): too few validation scans to keep every group, so split randomly
    stratify = groups
    num_val = int(len(indices) * val_fraction)
    if num_val < len(set(groups)):
        stratify = None

    train_indices, val_indices = train_test_split(
        indices, test_size=val_fraction, stratify=stratify, random_state=seed,
    )
    return train_indices, val_indices


def to_device(batch, device):
    """
    Move every tensor in a batch to the GPU (or CPU); keep other values (scan names).

    Args:
        batch: dict from the DataLoader
        device: "cuda" or "cpu"

    Returns:
        dict with the same keys, tensors on device
    """
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(model, batches, device, optimizer=None):
    """
    Go through all batches once and return the average loss.

      - optimizer given -> training: update the weights
      - no optimizer    -> validation: only measure

    Args:
        model: SilenceModel
        batches: DataLoader or list of batches
        device: "cuda" or "cpu"
        optimizer: optimizer, or None for validation

    Returns:
        average loss over the batches
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    num_batches = 0
    for batch in batches:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training):
            predicted = model(batch)
            loss = cosine_loss(predicted, batch["target"], batch["is_hidden"])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def main():
    config = load_config()
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    root = config["data_root"]
    dataset = SilenceDataset(root, config["split"], 
                             config["hide_prob"], 
                             config["seed"])
    train_indices, val_indices = split_train_val(dataset, 
                                                 config["val_fraction"], 
                                                 config["seed"])
    print(f"scans: {len(train_indices)} train, {len(val_indices)} validation")

    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=config["batch_size"], 
                              shuffle=True)
    # load validation batches once, so their hidden regions stay the same every epoch
    val_batches = list(DataLoader(Subset(dataset, val_indices), 
                                  batch_size=config["batch_size"]))

    model = SilenceModel(config["d_model"], 
                         config["num_heads"], 
                         config["num_layers"],
                         config["dropout"], 
                         config["geometry"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])

    # one folder per run, e.g. runs/regression_both
    run_dir = Path(root) / "runs" / f"{config['head']}_{config['geometry']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "val_loss", "val_cosine", "seconds"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_batches, device)
        seconds = time.time() - start

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val cosine {1 - val_loss:.4f}  ({seconds:.1f}s)")
        log.writerow([epoch, train_loss, val_loss, 1 - val_loss, seconds])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"no improvement for {config['patience']} epochs, stop")
                break

    log_file.close()
    print(f"best val cosine: {1 - best_val_loss:.4f}  saved to {run_dir}")


if __name__ == "__main__":
    main()