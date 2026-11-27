"""
Sentence retrieval

Turns a predicted text vector into a real doctor sentence: find the most similar
text vector (cosine) of the same region in the training set and return its text.
The bank uses training texts only, never test texts (no leakage).

Input:  predicted vectors (N, 768) for one region
Output: top-k texts, their states and cosine scores
"""

from pathlib import Path

import numpy as np
import pandas as pd

# pleura is merged into lung in the dataset, so pleura texts go to the lung bank
REGION_RENAME = {"pleura": "lung"}


class SentenceBank:
    """
    Written texts and their vectors, grouped by region.

    Built from text_vectors/sentences_{split}.csv and text_{split}.npy:
      - SILENT rows are skipped (no text)
      - pleura rows are added to lung
      - vectors are normalized, so a dot product = cosine similarity

    Args:
        root: data folder (data/ locally, silence_ct on Drive)
        split: which texts to use; "train" for real results
    """

    def __init__(self, root, split="train"):
        table = pd.read_csv(f"{root}/text_vectors/sentences_{split}.csv")
        vectors = np.load(f"{root}/text_vectors/text_{split}.npy").astype(np.float32)

        self.texts = {}       # {region: list of texts}
        self.states = {}      # {region: list of "ABNORMAL" / "NORMAL"}
        self.vectors = {}     # {region: (n, 768) normalized}

        rows_by_region = {}
        for i, row in enumerate(table.itertuples(index=False)):
            if row.state == "SILENT":
                continue
            region = REGION_RENAME.get(row.region, row.region)
            if region not in rows_by_region:
                rows_by_region[region] = []
            rows_by_region[region].append(i)

        for region, rows in rows_by_region.items():
            self.texts[region] = table["sentence"].iloc[rows].tolist()
            self.states[region] = table["state"].iloc[rows].tolist()
            self.vectors[region] = normalize(vectors[rows])

    def retrieve(self, predicted, region, top_k=1):
        """
        Find the top_k most similar texts of one region for each predicted vector.

        Args:
            predicted: (N, 768) predicted text vectors, all for the same region
            region: region name, e.g. "esophagus"
            top_k: how many texts to return per vector

        Returns:
            texts: list of N lists, each with top_k texts
            states: list of N lists, each with top_k states
            scores: (N, top_k) cosine similarity
        """
        similarity = normalize(predicted) @ self.vectors[region].T    # (N, n_bank)
        best = np.argsort(-similarity, axis=1)[:, :top_k]

        texts = []
        states = []
        for row in best:
            texts.append([self.texts[region][k] for k in row])
            states.append([self.states[region][k] for k in row])
        scores = np.take_along_axis(similarity, best, axis=1)
        return texts, states, scores


def normalize(vectors):
    """
    Scale each vector to length 1, so a dot product is a cosine similarity.

    Args:
        vectors: (N, D) array

    Returns:
        (N, D) array, each row with length 1
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-8)


if __name__ == "__main__":
    # PYTHONPATH=src python -m silence_ct.models.retrieval
    # a bank vector should find itself with cosine 1.0
    root = "/content/drive/MyDrive/silence_ct"
    split = "train"
    if not Path(root).exists():
        root, split = "data", "valid"       # local copy only has valid

    bank = SentenceBank(root, split)
    for region, vectors in bank.vectors.items():
        print(f"{region:22s} {len(vectors)} texts")

    region = "esophagus"
    texts, states, scores = bank.retrieve(bank.vectors[region][:1], 
                                          region, top_k=3)
    print("\nquery:", bank.texts[region][0])
    for text, state, score in zip(texts[0], states[0], scores[0]):
        print(f"  {score:.3f} {state:8s} {text[:80]}")