"""
Training

Trains SilenceModel to predict the text vectors of hidden regions. The 5,000 train
scans are split into train (90%) and validation (10%); the test set (1,564 valid
scans) is not used here. The model with the best mean validation ROC-AUC
(abnormal detection) is saved, with per-region thresholds chosen on validation.

Run: PYTHONPATH=src python -m silence_ct.train --config configs/config.yaml
Ablation example: ... --geometry none
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from silence_ct.data.dataset import SilenceDataset
from silence_ct.losses import cosine_loss, state_loss
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
    parser.add_argument("--name")        # run folder name; default from head and geometry
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
    config["name"] = args.name
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


def run_epoch(model, batches, device, state_weight, optimizer=None):
    """
    Go through all batches once and return the average losses.

      - optimizer given -> training: update the weights
      - no optimizer    -> validation: only measure
      - total loss = vector loss (cosine) + state_weight * state loss (cross-entropy)

    Args:
        model: SilenceModel
        batches: DataLoader or list of batches
        device: "cuda" or "cpu"
        state_weight: weight of the state loss
        optimizer: optimizer, or None for validation

    Returns:
        dict with average loss, vector_loss and state_loss
    """
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "vector_loss": 0.0, "state_loss": 0.0}
    num_batches = 0
    for batch in batches:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            vector_loss = cosine_loss(outputs["vector"], batch["target"], batch["is_hidden"])
            abnormal_loss = state_loss(outputs["abnormal_logit"], batch["state"], batch["is_hidden"])
            loss = vector_loss + state_weight * abnormal_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totals["loss"] += loss.item()
        totals["vector_loss"] += vector_loss.item()
        totals["state_loss"] += abnormal_loss.item()
        num_batches += 1

    averages = {}
    for key, value in totals.items():
        averages[key] = value / num_batches
    return averages


def predict_states(model, batches, device):
    """
    Predict P(abnormal) for every region of every scan, without updating the model.

    Args:
        model: SilenceModel
        batches: list of batches (validation)
        device: "cuda" or "cpu"

    Returns:
        probs: (N, 9) predicted P(abnormal)
        states: (N, 9) 1 ABNORMAL, 0 NORMAL, -1 SILENT
        is_hidden: (N, 9) True for hidden regions
    """
    model.eval()
    probs, states, is_hidden = [], [], []
    with torch.no_grad():
        for batch in batches:
            outputs = model(to_device(batch, device))
            probs.append(torch.sigmoid(outputs["abnormal_logit"]).cpu().numpy())
            states.append(batch["state"].numpy())
            is_hidden.append(batch["is_hidden"].numpy())
    return np.concatenate(probs), np.concatenate(states), np.concatenate(is_hidden)


def region_aucs(probs, states, is_hidden, regions):
    """
    ROC-AUC of "abnormal" for each region, on hidden regions only.

      - both classes present -> ROC-AUC
      - only one class       -> NaN (e.g. breast in a small validation set)

    Args:
        probs: (N, 9) predicted P(abnormal)
        states: (N, 9) true states
        is_hidden: (N, 9) True for hidden regions
        regions: list of the 9 region names

    Returns:
        dict {region: ROC-AUC}
    """
    aucs = {}
    for k, region in enumerate(regions):
        rows = is_hidden[:, k]
        labels = (states[rows, k] == 1).astype(int)
        if len(set(labels)) < 2:
            aucs[region] = float("nan")
        else:
            aucs[region] = roc_auc_score(labels, probs[rows, k])
    return aucs


def choose_thresholds(probs, states, is_hidden, regions):
    """
    Pick the P(abnormal) threshold with the best F1 for each region, on validation.

      - tries thresholds 0.05, 0.10, ..., 0.95
      - no abnormal case in validation -> keep 0.5

    Args:
        probs: (N, 9) predicted P(abnormal)
        states: (N, 9) true states
        is_hidden: (N, 9) True for hidden regions
        regions: list of the 9 region names

    Returns:
        dict {region: threshold}
    """
    thresholds = {}
    for k, region in enumerate(regions):
        rows = is_hidden[:, k]
        labels = (states[rows, k] == 1).astype(int)
        thresholds[region] = 0.5
        if labels.sum() == 0:
            continue

        best_f1 = -1.0
        for threshold in np.arange(0.05, 0.96, 0.05):
            predicted = (probs[rows, k] >= threshold).astype(int)
            f1 = f1_score(labels, predicted, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[region] = round(float(threshold), 2)
    return thresholds


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

    # one folder per run, e.g. runs/regression_both_state
    run_name = config["name"]
    if run_name is None:
        run_name = f"{config['head']}_{config['geometry']}"
        if config["state_weight"] > 0:
            run_name += "_state"
    run_dir = Path(root) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "train_vector_loss", "train_state_loss",
                  "val_loss", "val_vector_loss", "val_state_loss", "val_cosine",
                  "val_auc_heart", "val_auc_esophagus", "val_auc_mean", "seconds"])

    # best model = highest mean validation ROC-AUC (our main goal: abnormal detection)
    best_auc = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        train = run_epoch(model, train_loader, device, config["state_weight"], optimizer)
        val = run_epoch(model, val_batches, device, config["state_weight"])
        probs, states, is_hidden = predict_states(model, val_batches, device)
        aucs = region_aucs(probs, states, is_hidden, dataset.regions)
        mean_auc = float(np.nanmean(list(aucs.values())))
        seconds = time.time() - start

        if np.isnan(val["loss"]):
            raise ValueError("validation loss is NaN, check the model or data")

        val_cosine = 1 - val["vector_loss"]
        print(f"epoch {epoch:3d}  train {train['loss']:.4f}  val {val['loss']:.4f}  "
              f"cosine {val_cosine:.4f}  AUC heart {aucs['heart']:.3f}  "
              f"esophagus {aucs['esophagus']:.3f}  mean {mean_auc:.3f}  ({seconds:.1f}s)")
        log.writerow([epoch, train["loss"], train["vector_loss"], train["state_loss"],
                      val["loss"], val["vector_loss"], val["state_loss"], val_cosine,
                      aucs["heart"], aucs["esophagus"], mean_auc, seconds])
        log_file.flush()

        # epoch 1 is always saved, so a file exists even if AUC is NaN (tiny local test)
        if epoch == 1 or mean_auc > best_auc:
            best_auc = mean_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"no improvement for {config['patience']} epochs, stop")
                break
    log_file.close()

    # thresholds per region, chosen on validation with the best model
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    probs, states, is_hidden = predict_states(model, val_batches, device)
    thresholds = choose_thresholds(probs, states, is_hidden, dataset.regions)
    with open(run_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"best val mean AUC: {best_auc:.4f}  saved to {run_dir}")
    print("thresholds:", thresholds)

if __name__ == "__main__":
    main()
