"""
Diffusion head training (stage 2)

Trains a DiffusionHead on top of a trained, FROZEN SilenceModel (stage 1, e.g.
runs/regression_none_state). The stage-1 encoder gives the condition for each region,
and its state head (abnormality) is unchanged, so only the text-vector part changes.
Same data split as stage 1. Loss: noise-prediction MSE on hidden regions; validation
uses fixed noise so epochs can be compared. Saves diffusion_head.pt in runs/<name>.

Run: PYTHONPATH=src python -m silence_ct.train_diffusion --base_run <stage-1 run folder>
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from silence_ct.data.dataset import SilenceDataset
from silence_ct.evaluate import load_model
from silence_ct.models.diffusion_head import DiffusionHead
from silence_ct.train import split_train_val, to_device


def hidden_pairs(model, batch):
    """
    Condition and target of the hidden regions in one batch (stage-1 model frozen).

    Args:
        model: frozen SilenceModel
        batch: batch on the device

    Returns:
        condition: (N, d_model) encoded tokens of hidden regions
        target: (N, 768) their true text vectors
    """
    with torch.no_grad():
        encoded = model.encode(batch)
    hidden = batch["is_hidden"]
    return encoded[hidden], batch["target"][hidden]


def run_epoch(model, head, batches, device, optimizer=None):
    """
    Go through all batches once and return the average diffusion loss.

      - optimizer given -> training: random noise, update the head
      - no optimizer    -> validation: fixed noise (seed 0), only measure

    Args:
        model: frozen SilenceModel
        head: DiffusionHead
        batches: DataLoader or list of batches
        device: "cuda" or "cpu"
        optimizer: optimizer, or None for validation

    Returns:
        average loss
    """
    training = optimizer is not None
    head.train(training)
    generator = None
    if not training:
        generator = torch.Generator(device=device).manual_seed(0)

    total_loss = 0.0
    num_batches = 0
    for batch in batches:
        batch = to_device(batch, device)
        condition, target = hidden_pairs(model, batch)
        with torch.set_grad_enabled(training):
            loss = head.loss(condition, target, generator)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run", required=True)   # stage-1 run folder
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--root")                       # data folder; default from base run
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--name")                       # default: diffusion_<base run settings>
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)
    if args.epochs:
        settings["diffusion_epochs"] = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, base_config, _ = load_model(args.base_run, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    torch.manual_seed(base_config["seed"])

    # same data and split as stage 1
    root = args.root or base_config["data_root"]
    dataset = SilenceDataset(root, base_config["split"], base_config["hide_prob"], base_config["seed"])
    train_indices, val_indices = split_train_val(dataset, base_config["val_fraction"], base_config["seed"])
    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=base_config["batch_size"], shuffle=True)
    val_batches = list(DataLoader(Subset(dataset, val_indices), batch_size=base_config["batch_size"]))
    print(f"device: {device}  scans: {len(train_indices)} train, {len(val_indices)} validation")

    head = DiffusionHead(base_config["d_model"], width=settings["diffusion_width"],
                         num_blocks=settings["diffusion_blocks"],
                         num_steps=settings["diffusion_steps"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["diffusion_lr"])

    # run folder, e.g. runs/diffusion_none_state
    run_name = args.name or "diffusion_" + Path(args.base_run).name.replace("regression_", "")
    run_dir = Path(root) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    diffusion_config = {"base_run": str(args.base_run)}
    for key in ["diffusion_width", "diffusion_blocks", "diffusion_steps", "sample_steps",
                "num_samples", "diffusion_lr", "diffusion_epochs", "diffusion_patience"]:
        diffusion_config[key] = settings[key]
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(diffusion_config, f)

    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "val_loss", "seconds"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, settings["diffusion_epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, head, train_loader, device, optimizer)
        val_loss = run_epoch(model, head, val_batches, device)
        seconds = time.time() - start

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  ({seconds:.1f}s)")
        log.writerow([epoch, train_loss, val_loss, seconds])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(head.state_dict(), run_dir / "diffusion_head.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings["diffusion_patience"]:
                print(f"no improvement for {settings['diffusion_patience']} epochs, stop")
                break

    log_file.close()
    print(f"best val loss: {best_val_loss:.4f}  saved to {run_dir}")


if __name__ == "__main__":
    main()
