"""
Evaluation

Scores a trained model on the test set (RadGenome valid split, 1,564 scans) and
compares it with two baselines that need no training. Only hidden written regions
are scored, because silent regions have no true text.

    mean vector:      always predict the average train text vector of the region
    silence = normal: always predict the most common NORMAL train sentence of the region

For every method and region: cosine with the true vector, and retrieval from the
train bank -> normal/abnormal accuracy, precision, recall, F1 and ROC-AUC.

Run: PYTHONPATH=src python -m silence_ct.evaluate --run_dir <path to run folder>
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from silence_ct.data.dataset import STATE_IDS, SilenceDataset
from silence_ct.models.model import SilenceModel
from silence_ct.models.retrieval import SentenceBank, normalize
from silence_ct.train import to_device


def load_model(run_dir, device):
    """
    Rebuild the model from the run's config and load its best weights.

    Args:
        run_dir: run folder with config.yaml and best.pt
        device: "cuda" or "cpu"

    Returns:
        model: SilenceModel in evaluation mode
        config: dict of settings used for training
    """
    with open(Path(run_dir) / "config.yaml") as f:
        config = yaml.safe_load(f)

    model = SilenceModel(config["d_model"], 
                         config["num_heads"], 
                         config["num_layers"],
                         config["dropout"], 
                         config["geometry"])
    
    state = torch.load(Path(run_dir) / "best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def collect_predictions(model, dataset, device, batch_size):
    """
    Run the model once over the whole dataset and keep what is needed for scoring.

    The dataset is read once in order, so the hidden regions are fixed by its seed:
      - same seed -> same hidden regions for every model (fair comparison)

    Args:
        model: SilenceModel in evaluation mode
        dataset: SilenceDataset of the test split
        device: "cuda" or "cpu"
        batch_size: scans per batch

    Returns:
        dict of numpy arrays: scans (N,), predicted (N, 9, 768), target (N, 9, 768),
        is_hidden (N, 9), state (N, 9)
    """
    scans, predicted, target, is_hidden, state = [], [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            scans.extend(batch["scan"])
            target.append(batch["target"].numpy())
            is_hidden.append(batch["is_hidden"].numpy())
            state.append(batch["state"].numpy())

            batch = to_device(batch, device)
            predicted.append(model(batch).cpu().numpy())

    return {
        "scans": np.array(scans),
        "predicted": np.concatenate(predicted),
        "target": np.concatenate(target),
        "is_hidden": np.concatenate(is_hidden),
        "state": np.concatenate(state),
    }


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


def score_method(predicted, true_vectors, true_states, bank, region, top_k):
    """
    Score one method on the hidden regions of one region type.

      - cosine: predicted vs true text vector
      - top-1 retrieved state -> accuracy, precision, recall, F1 (ABNORMAL = positive)
      - share of ABNORMAL among top_k retrieved -> score for ROC-AUC
        (only one class in the test rows -> ROC-AUC is NaN)

    Args:
        predicted: (N, 768) predicted vectors
        true_vectors: (N, 768) true text vectors
        true_states: (N,) 1 ABNORMAL, 0 NORMAL
        bank: SentenceBank built from train texts
        region: region name
        top_k: number of retrieved texts used for the ROC-AUC score

    Returns:
        scores: dict of metrics
        retrieved: list of N top-1 texts
    """
    cosine = np.sum(normalize(predicted) * normalize(true_vectors), axis=1)

    texts, states, _ = bank.retrieve(predicted, region, top_k=top_k)
    predicted_states = []
    abnormal_share = []
    for row in states:
        predicted_states.append(1 if row[0] == "ABNORMAL" else 0)
        abnormal_share.append(row.count("ABNORMAL") / len(row))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_states, predicted_states, 
        average="binary", pos_label=1, zero_division=0,
    )
    roc_auc = float("nan")
    if len(set(true_states)) == 2:
        roc_auc = roc_auc_score(true_states, abnormal_share)

    scores = {
        "n": len(true_states),
        "n_abnormal": int(sum(true_states)),
        "cosine": float(cosine.mean()),
        "accuracy": accuracy_score(true_states, predicted_states),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    retrieved = [row[0] for row in texts]
    return scores, retrieved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root")                       # data folder; default from config
    parser.add_argument("--split", default="valid")     # RadGenome valid = our test set
    parser.add_argument("--bank_split", default="train")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.run_dir, device)
    root = args.root or config["data_root"]

    dataset = SilenceDataset(root, args.split, 
                             config["hide_prob"], 
                             config["seed"])
    outputs = collect_predictions(model, 
                                  dataset, 
                                  device, 
                                  config["batch_size"])
    bank = SentenceBank(root, args.bank_split)
    print(f"{args.split}: {len(dataset)} scans, bank: {args.bank_split}")

    results = []
    examples = []
    for k, region in enumerate(dataset.regions):
        # hidden written regions of this type: the only ones with a true answer
        rows = outputs["is_hidden"][:, k]
        if rows.sum() == 0 or region not in bank.vectors:
            continue
        true_vectors = outputs["target"][rows, k]
        true_states = (outputs["state"][rows, k] == STATE_IDS["ABNORMAL"]).astype(int)

        methods = {"model": outputs["predicted"][rows, k]}
        for name, vector in baseline_vectors(bank, region).items():
            methods[name] = np.tile(vector, (len(true_states), 1))

        for name, predicted in methods.items():
            scores, retrieved = score_method(predicted, true_vectors, true_states,
                                             bank, region, args.top_k)
            results.append({"method": name, "region": region, **scores})

            if name == "model":
                for scan, state, text in zip(outputs["scans"][rows], true_states, retrieved):
                    examples.append({"scan": scan, "region": region,
                                     "true_state": state, "retrieved": text})

    results = pd.DataFrame(results)
    results.to_csv(Path(args.run_dir) / f"eval_{args.split}.csv", index=False)
    pd.DataFrame(examples).to_csv(Path(args.run_dir) / f"examples_{args.split}.csv", index=False)

    pd.set_option("display.width", 200)
    for region in ["heart", "esophagus"]:
        print(f"\n{region}")
        print(results[results["region"] == region].drop(columns="region")
              .round(3).to_string(index=False))
    print(f"\nall regions saved to {Path(args.run_dir) / f'eval_{args.split}.csv'}")


# TODO: text scores (BLEU / ROUGE / BERTScore) between retrieved and true sentence
# TODO: predictions for SILENT regions (no answer; report predicted state distribution)


if __name__ == "__main__":
    main()