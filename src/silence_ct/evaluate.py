"""
Evaluation

Scores a trained model on the test set (RadGenome valid split, 1,564 scans) and
compares it with two baselines that need no training.

Leave-one-region-out: every WRITTEN region of every scan is predicted with only that
region hidden (all other written regions visible). Silent regions have no true text,
so they are not scored. No random hiding, so results are the same every run.

    model:             state from the state head (per-region threshold), sentence from retrieval
    model (retrieval): state from the top-1 retrieved sentence
    model (diffusion samples): diffusion runs only; P(abnormal) = share of the samples
                       whose retrieved sentence is abnormal
    mean vector:       always predict the average train text vector of the region
    silence = normal:  always predict the most common NORMAL train sentence of the region

For every method and region: cosine with the true vector, normal/abnormal accuracy,
precision, recall, F1, ROC-AUC with a bootstrap 95% CI, plus a macro average over
regions. Saves ROC curves (heart + esophagus, and all regions).
For a diffusion run (folder with diffusion_head.pt), the stage-1 model named in its
config is loaded; the text vector is the mean of the samples, and the examples file
also has the sample spread (uncertainty: 0 = all samples agree).

Run: PYTHONPATH=src python -m silence_ct.evaluate --run_dir <path to run folder>
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from silence_ct.data.dataset import STATE_IDS, SilenceDataset
from silence_ct.models.diffusion_head import DiffusionHead
from silence_ct.models.model import SilenceModel
from silence_ct.models.retrieval import SentenceBank, normalize
from silence_ct.plot import MUTED_COLOR, STYLE, TRAIN_COLOR
from silence_ct.train import to_device

# ROC curve colour per region (others use the default blue)
REGION_COLORS = {"heart": "#6a4ce2", "esophagus": "#00b894"}

SCORE_COLUMNS = ["cosine", "accuracy", "precision", "recall", "f1", "roc_auc"]


def load_model(run_dir, device):
    """
    Rebuild the model from the run's config and load its best weights.

      - run with a state head    -> all weights loaded
      - older run without one    -> state head left untrained and NOT used
                                    (abnormal score comes from retrieval instead)

    Args:
        run_dir: run folder with config.yaml and best.pt
        device: "cuda" or "cpu"

    Returns:
        model: SilenceModel in evaluation mode
        config: dict of settings used for training
        has_state_head: True if the run trained a state head
    """
    with open(Path(run_dir) / "config.yaml") as f:
        config = yaml.safe_load(f)

    model = SilenceModel(config["d_model"], config["num_heads"], config["num_layers"],
                         config["dropout"], config["geometry"])
    state = torch.load(Path(run_dir) / "best.pt", map_location=device)
    has_state_head = "state_head.weight" in state
    model.load_state_dict(state, strict=has_state_head)
    model.to(device).eval()
    return model, config, has_state_head


def make_batch(dataset, samples):
    """
    Build one model batch where each sample hides exactly one written region.

      - hidden region   -> text zeros, has_text False (as in training)
      - other written   -> text visible
      - silent regions  -> no text (as in training)

    Args:
        dataset: SilenceDataset (cached image, geometry and text)
        samples: list of (scan name, region index)

    Returns:
        dict of tensors in the SilenceDataset batch format (image, has_image, text,
        has_text, distance, touching), plus scan (list), region (B,),
        true_vector (B, 768) and true_state (B,) of the hidden region
    """
    image, has_image, text, has_text, distance, touching = [], [], [], [], [], []
    scans, region, true_vector, true_state = [], [], [], []

    for scan, k in samples:
        row = dataset.row[scan]
        scan_text, state = dataset.text[scan]

        visible_text = scan_text.copy()
        visible_text[k] = 0
        visible = state != STATE_IDS["SILENT"]
        visible[k] = False

        image.append(dataset.image[row])
        has_image.append(dataset.has_image[row])
        text.append(visible_text)
        has_text.append(visible)
        distance.append(dataset.distance[row])
        touching.append(dataset.touching[row])

        scans.append(scan)
        region.append(k)
        true_vector.append(scan_text[k])
        true_state.append(state[k])

    return {
        "scan": scans,
        "image": torch.from_numpy(np.stack(image)),
        "has_image": torch.from_numpy(np.stack(has_image)),
        "text": torch.from_numpy(np.stack(text)),
        "has_text": torch.from_numpy(np.stack(has_text)),
        "distance": torch.from_numpy(np.stack(distance)),
        "touching": torch.from_numpy(np.stack(touching)),
        "region": torch.tensor(region),
        "true_vector": torch.from_numpy(np.stack(true_vector)),
        "true_state": torch.tensor(true_state),
    }


def leave_one_out_predictions(model, dataset, device, batch_size=256, head=None,
                              num_samples=10, sample_steps=50, generator=None):
    """
    Predict every written region of every scan with only that region hidden.

      - head given (diffusion run) -> also draw num_samples text vectors per region

    Args:
        model: SilenceModel in evaluation mode
        dataset: SilenceDataset of the test split
        device: "cuda" or "cpu"
        batch_size: samples per batch
        head: DiffusionHead, or None
        num_samples: diffusion samples per region
        sample_steps: denoising steps per sample
        generator: random generator for the diffusion start noise

    Returns:
        dict of numpy arrays, one row per (scan, written region):
        scans (M,), region (M,) index 0..8, predicted (M, 768), abnormal_prob (M,),
        true_vector (M, 768), true_state (M,), and samples (M, num_samples, 768) if head
    """
    samples = []
    for scan in dataset.scans:
        _, state = dataset.text[scan]
        for k in range(len(state)):
            if state[k] != STATE_IDS["SILENT"]:
                samples.append((scan, k))

    outputs = {"scans": [], "region": [], "predicted": [], "abnormal_prob": [],
               "true_vector": [], "true_state": [], "samples": []}
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = make_batch(dataset, samples[start:start + batch_size])
            result = model(to_device(batch, device))

            # keep only the prediction for the hidden region of each sample
            index = torch.arange(len(batch["scan"]), device=device)
            region = batch["region"].to(device)
            vector = result["vector"][index, region]
            prob = torch.sigmoid(result["abnormal_logit"][index, region])

            outputs["scans"].extend(batch["scan"])
            outputs["region"].append(batch["region"].numpy())
            outputs["predicted"].append(vector.cpu().numpy())
            outputs["abnormal_prob"].append(prob.cpu().numpy())
            outputs["true_vector"].append(batch["true_vector"].numpy())
            outputs["true_state"].append(batch["true_state"].numpy())

            if head is not None:
                condition = result["encoded"][index, region]
                drawn = head.sample(condition, num_samples, sample_steps, generator)
                outputs["samples"].append(drawn.cpu().numpy())

    arrays = {"scans": np.array(outputs["scans"])}
    keys = ["region", "predicted", "abnormal_prob", "true_vector", "true_state"]
    if head is not None:
        keys.append("samples")
    for key in keys:
        arrays[key] = np.concatenate(outputs[key])
    return arrays


def sample_spread(samples):
    """
    How much the diffusion samples of one region disagree (uncertainty).

      - all samples point the same way -> 0
      - samples unrelated to each other -> close to 1

    Args:
        samples: (N, K, 768) samples, K per region

    Returns:
        spread: (N,) 1 - mean cosine between different samples of the same region
    """
    normalized = samples / np.linalg.norm(samples, axis=2, keepdims=True)
    gram = np.einsum("nkd,njd->nkj", normalized, normalized)
    num_samples = samples.shape[1]
    mean_cosine = (gram.sum(axis=(1, 2)) - num_samples) / (num_samples * (num_samples - 1))
    return 1 - mean_cosine


def baseline_vectors(bank, region):
    """
    Build the two no-training baseline vectors for one region from the train bank.

      - mean vector      -> average of all written train vectors of the region
      - silence = normal -> vector of the most common NORMAL train sentence

    Args:
        bank: SentenceBank built from train texts
        region: region name

    Returns:
        dict {"mean vector": (768,), "silence = normal": (768,)}
    """
    mean_vector = bank.vectors[region].mean(axis=0)

    normal_texts = []
    for text, state in zip(bank.texts[region], bank.states[region]):
        if state == "NORMAL":
            normal_texts.append(text)
    most_common_normal = Counter(normal_texts).most_common(1)[0][0]
    normal_vector = bank.vectors[region][bank.texts[region].index(most_common_normal)]

    return {"mean vector": mean_vector, "silence = normal": normal_vector}


def retrieve_states(predicted, bank, region, top_k):
    """
    Turn predicted vectors into states and sentences by retrieval from the train bank.

      - top-1 retrieved state -> predicted state (1 ABNORMAL, 0 NORMAL)
      - share of ABNORMAL among top_k -> abnormal score for ROC-AUC

    Args:
        predicted: (N, 768) predicted vectors
        bank: SentenceBank built from train texts
        region: region name
        top_k: number of retrieved texts used for the score

    Returns:
        predicted_states: (N,) 0/1
        abnormal_share: (N,) between 0 and 1
        retrieved: list of N top-1 texts
    """
    texts, states, _ = bank.retrieve(predicted, region, top_k=top_k)
    predicted_states = []
    abnormal_share = []
    for row in states:
        predicted_states.append(1 if row[0] == "ABNORMAL" else 0)
        abnormal_share.append(row.count("ABNORMAL") / len(row))

    retrieved = [row[0] for row in texts]
    return np.array(predicted_states), np.array(abnormal_share), retrieved


def bootstrap_auc(labels, scores, num_resamples=1000, seed=0):
    """
    95% confidence interval of ROC-AUC by resampling the test cases with replacement.

      - same score for every case (baselines) -> (0.5, 0.5)
      - only one class in the test cases      -> (NaN, NaN)
      - a resample with only one class        -> skipped

    Args:
        labels: (N,) 1 ABNORMAL, 0 NORMAL
        scores: (N,) higher = more likely abnormal
        num_resamples: number of bootstrap resamples
        seed: random seed, so the interval is the same every run

    Returns:
        (low, high): 2.5th and 97.5th percentile of the resampled AUCs
    """
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    if np.all(scores == scores[0]):
        return 0.5, 0.5

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(num_resamples):
        pick = rng.integers(0, len(labels), len(labels))
        if len(set(labels[pick])) < 2:
            continue
        aucs.append(roc_auc_score(labels[pick], scores[pick]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def score_method(method, region, predicted, true_vectors, true_states,
                 predicted_states, abnormal_score):
    """
    Compute all scores of one method on the written regions of one region type.

      - cosine between predicted and true text vectors
      - accuracy, precision, recall, F1 with ABNORMAL as the positive class
      - ROC-AUC from the abnormal score, with a bootstrap 95% CI

    Args:
        method: method name
        region: region name
        predicted: (N, 768) predicted vectors
        true_vectors: (N, 768) true text vectors
        true_states: (N,) 1 ABNORMAL, 0 NORMAL
        predicted_states: (N,) 0/1
        abnormal_score: (N,) higher = more likely abnormal

    Returns:
        dict with one row of results
    """
    cosine = np.sum(normalize(predicted) * normalize(true_vectors), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_states, predicted_states, average="binary", pos_label=1, zero_division=0,
    )
    roc_auc = float("nan")
    if len(set(true_states)) == 2:
        roc_auc = roc_auc_score(true_states, abnormal_score)
    auc_low, auc_high = bootstrap_auc(true_states, abnormal_score)

    return {
        "method": method,
        "region": region,
        "n": len(true_states),
        "n_abnormal": int(sum(true_states)),
        "cosine": float(cosine.mean()),
        "accuracy": accuracy_score(true_states, predicted_states),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "auc_low": auc_low,
        "auc_high": auc_high,
    }


def macro_average(results):
    """
    Average each score over regions, per method (every region counts the same).

    Args:
        results: table with one row per method and region

    Returns:
        table with one "macro average" row per method (NaN scores are skipped)
    """
    rows = []
    for method, group in results.groupby("method", sort=False):
        row = {"method": method, "region": "macro average",
               "n": int(group["n"].sum()), "n_abnormal": int(group["n_abnormal"].sum())}
        for column in SCORE_COLUMNS:
            row[column] = group[column].mean()
        row["auc_low"] = float("nan")
        row["auc_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc(roc_data, out_path, columns=2):
    """
    Draw ROC curves of the model's P(abnormal) on the test set, one panel per region.

      - curve above the diagonal -> better than guessing
      - dashed diagonal          -> random guessing (AUC 0.5); the baselines sit here,
                                    because they give every scan the same answer

    Args:
        roc_data: dict {region: (true_states (N,), abnormal_prob (N,))}
        out_path: where to save the figure
        columns: panels per row (2 for the main figure, 3 for all regions)
    """
    num_panels = len(roc_data)
    num_columns = min(columns, num_panels)
    num_rows = math.ceil(num_panels / num_columns)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(num_rows, num_columns, squeeze=False,
                                 figsize=(4.2 * num_columns, 4.2 * num_rows))
        axes = axes.flatten()

        for ax, (region, (labels, probs)) in zip(axes, roc_data.items()):
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, probs)
            auc = roc_auc_score(labels, probs)
            color = REGION_COLORS.get(region, TRAIN_COLOR)

            ax.plot(false_positive_rate, true_positive_rate, color=color,
                    label=f"model (AUC {auc:.3f})")
            ax.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1,
                    label="random / baselines (AUC 0.5)")

            ax.set_title(f"{region} (n = {len(labels)}, abnormal = {int(sum(labels))})",
                         fontsize=10)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_aspect("equal")
            ax.legend(loc="lower right", fontsize=8)

        # hide empty panels in the grid
        for ax in axes[num_panels:]:
            ax.set_visible(False)

        fig.suptitle(Path(out_path).parent.name, x=0.01, ha="left", color=MUTED_COLOR, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root")                       # data folder; default from config
    parser.add_argument("--split", default="valid")     # RadGenome valid = our test set
    parser.add_argument("--bank_split", default="train")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)

    # diffusion run: its config names the frozen stage-1 run
    head = None
    diffusion_config = None
    base_dir = run_dir
    if (run_dir / "diffusion_head.pt").exists():
        with open(run_dir / "config.yaml") as f:
            diffusion_config = yaml.safe_load(f)
        base_dir = Path(diffusion_config["base_run"])

    model, config, has_state_head = load_model(base_dir, device)
    root = args.root or config["data_root"]

    if diffusion_config is not None:
        head = DiffusionHead(config["d_model"], width=diffusion_config["diffusion_width"],
                             num_blocks=diffusion_config["diffusion_blocks"],
                             num_steps=diffusion_config["diffusion_steps"])
        head.load_state_dict(torch.load(run_dir / "diffusion_head.pt", map_location=device))
        head.to(device).eval()

    # per-region thresholds chosen on validation by train.py (0.5 if missing)
    thresholds = {}
    if (base_dir / "thresholds.json").exists():
        with open(base_dir / "thresholds.json") as f:
            thresholds = json.load(f)

    dataset = SilenceDataset(root, args.split, config["hide_prob"], config["seed"])
    if head is None:
        outputs = leave_one_out_predictions(model, dataset, device)
    else:
        generator = torch.Generator(device=device).manual_seed(0)
        outputs = leave_one_out_predictions(model, dataset, device, head=head,
                                            num_samples=diffusion_config["num_samples"],
                                            sample_steps=diffusion_config["sample_steps"],
                                            generator=generator)
        # text vector of a diffusion run = mean of its samples
        outputs["predicted"] = normalize(outputs["samples"].mean(axis=1))

    bank = SentenceBank(root, args.bank_split)
    print(f"{args.split}: {len(dataset)} scans, {len(outputs['scans'])} written regions "
          f"scored (leave-one-region-out), bank: {args.bank_split}")

    results = []
    examples = []
    roc_data = {}
    for k, region in enumerate(dataset.regions):
        rows = outputs["region"] == k
        if rows.sum() == 0 or region not in bank.vectors:
            continue
        true_vectors = outputs["true_vector"][rows]
        true_states = (outputs["true_state"][rows] == STATE_IDS["ABNORMAL"]).astype(int)

        model_vectors = outputs["predicted"][rows]
        states, share, retrieved = retrieve_states(model_vectors, bank, region, args.top_k)

        if has_state_head:
            # model: state from the state head, sentence from retrieval
            model_prob = outputs["abnormal_prob"][rows]
            model_states = (model_prob >= thresholds.get(region, 0.5)).astype(int)
            results.append(score_method("model", region, model_vectors, true_vectors,
                                        true_states, model_states, model_prob))
        else:
            # older run without a state head: abnormal score from retrieval
            model_prob = share
        if len(set(true_states)) == 2:
            roc_data[region] = (true_states, model_prob)

        # model (retrieval): state from the top-1 retrieved sentence
        results.append(score_method("model (retrieval)", region, model_vectors, true_vectors,
                                    true_states, states, share))

        # model (diffusion samples): each sample -> top-1 sentence -> share of abnormal
        spread = None
        if head is not None:
            samples = outputs["samples"][rows]
            num_samples = samples.shape[1]
            _, sample_states, _ = bank.retrieve(samples.reshape(-1, samples.shape[2]),
                                                region, top_k=1)
            sample_abnormal = []
            for row in sample_states:
                sample_abnormal.append(row[0] == "ABNORMAL")
            sample_share = np.array(sample_abnormal).reshape(-1, num_samples).mean(axis=1)
            results.append(score_method("model (diffusion samples)", region, model_vectors,
                                        true_vectors, true_states,
                                        (sample_share >= 0.5).astype(int), sample_share))
            spread = sample_spread(samples)

        for i, (scan, state, prob, text) in enumerate(zip(outputs["scans"][rows], true_states,
                                                          model_prob, retrieved)):
            example = {"scan": scan, "region": region, "true_state": state,
                       "abnormal_prob": prob, "retrieved": text}
            if spread is not None:
                example["sample_spread"] = spread[i]
            examples.append(example)

        # baselines: the same vector for every scan
        for name, vector in baseline_vectors(bank, region).items():
            vectors = np.tile(vector, (len(true_states), 1))
            states, share, _ = retrieve_states(vectors, bank, region, args.top_k)
            results.append(score_method(name, region, vectors, true_vectors,
                                        true_states, states, share))

    results = pd.DataFrame(results)
    results = pd.concat([results, macro_average(results)], ignore_index=True)
    results.to_csv(run_dir / f"eval_{args.split}.csv", index=False)
    examples = pd.DataFrame(examples)
    examples.to_csv(run_dir / f"examples_{args.split}.csv", index=False)

    pd.set_option("display.width", 220)
    for region in ["heart", "esophagus", "macro average"]:
        print(f"\n{region}")
        print(results[results["region"] == region].drop(columns="region")
              .round(3).to_string(index=False))

    main_method = "model" if has_state_head else "model (retrieval)"
    print(f"\n{main_method}, all regions")
    model_rows = results[results["method"] == main_method]
    print(model_rows[["region", "n", "n_abnormal", "f1", "roc_auc", "auc_low", "auc_high"]]
          .round(3).to_string(index=False))

    if head is not None:
        # uncertainty: is the model less sure about abnormal regions?
        print("\nmean sample spread (uncertainty) by true state, heart and esophagus")
        main_examples = examples[examples["region"].isin(["heart", "esophagus"])]
        print(main_examples.groupby(["region", "true_state"])["sample_spread"]
              .mean().round(4).to_string())

    # ROC figures: main organs (paper) and all regions (appendix)
    main_regions = {}
    for region in ["heart", "esophagus"]:
        if region in roc_data:
            main_regions[region] = roc_data[region]
    if main_regions:
        plot_roc(main_regions, run_dir / f"roc_{args.split}.png", columns=2)
    if roc_data:
        plot_roc(roc_data, run_dir / f"roc_all_regions_{args.split}.png", columns=3)
    print(f"\nsaved to {run_dir}: eval_{args.split}.csv, examples_{args.split}.csv, "
          f"roc_{args.split}.png, roc_all_regions_{args.split}.png")


# TODO: text scores (BLEU / ROUGE / BERTScore) between retrieved and true sentence
# TODO: predictions for SILENT regions (no answer; report predicted state distribution)


if __name__ == "__main__":
    main()
