"""
Dataset for silence-aware report completion

One item = one CT scan with 9 regions (pleura merged into lung). For each region:
image vector (OrganLens), text vector (CT-CLIP), state (abnormal / normal / silent),
and geometry (distance, touching) between regions.

Some written regions are hidden; the model predicts their text vector.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# OrganLens output order (from organ_vectors/organlens_settings.json)
ORGANLENS_ORGANS = ["heart", "lung", "aorta", "esophagus", "trachea", "liver",
                    "spleen", "kidneys", "pancreas", "stomach", "intestine"]

# which OrganLens organs describe each region; several organs are averaged, none = no image
REGION_TO_ORGANS = {
    "abdomen": ["liver", "spleen", "kidneys", "pancreas", "stomach", "intestine"],
    "bone": [],
    "breast": [],
    "esophagus": ["esophagus"],
    "heart": ["heart"],
    "lung": ["lung"],
    "mediastinum": ["aorta"],
    "thyroid": [],
    "trachea and bronchie": ["trachea"],
}

STATE_IDS = {"ABNORMAL": 1, "NORMAL": 0, "SILENT": -1}

# used when a region mask is empty, so its center (and distances) are NaN
FAR_DISTANCE_MM = 300.0


def organs_to_regions(organ_vectors, regions):
    """
    Turn OrganLens organ vectors into one vector per region.

    OrganLens gives 11 organs, but our masks and report use 9 regions, so we map them
    with REGION_TO_ORGANS:
      - one organ  -> copy it            (heart -> heart, aorta -> mediastinum)
      - many organs -> average them      (liver, spleen, ... -> abdomen)
      - no organ   -> zeros, has_image = False   (bone, breast, thyroid)

    Args:
        organ_vectors: (11, 1024) array, rows in ORGANLENS_ORGANS order
        regions: list of the 9 region names, in the order the model uses

    Returns:
        image: (9, 1024) one vector per region
        has_image: (9,) True if the region has an OrganLens vector
    """
    image = np.zeros((len(regions), organ_vectors.shape[1]), dtype=np.float32)
    has_image = np.zeros(len(regions), dtype=bool)

    for k, region in enumerate(regions):
        organs = REGION_TO_ORGANS[region]
        if not organs:
            continue
        rows = [ORGANLENS_ORGANS.index(organ) for organ in organs]
        image[k] = organ_vectors[rows].mean(axis=0)
        has_image[k] = True

    return image, has_image


def merge_lung_and_pleura(lung, pleura):
    """
    Combine the lung and pleura rows of one scan into one lung entry.

    In RadGenome the pleura mask is the same as the lung mask, and "pleura" sentences
    are often about the lung, so we treat them as one region:
      - both written  -> average the two vectors; ABNORMAL if either is abnormal
      - only pleura   -> use the pleura vector and state
      - otherwise     -> keep the lung vector and state

    Args:
        lung: dict with "vector" (768,) and "state" ("ABNORMAL" / "NORMAL" / "SILENT")
        pleura: same, for the pleura row

    Returns:
        vector: (768,) text vector for the merged lung
        state: state of the merged lung
    """
       
    lung_written = lung["state"] != "SILENT"
    pleura_written = pleura["state"] != "SILENT"

    if lung_written and pleura_written:
        vector = (lung["vector"] + pleura["vector"]) / 2
        if "ABNORMAL" in (lung["state"], pleura["state"]):
            state = "ABNORMAL"
        else:
            state = "NORMAL"
        return vector, state

    if pleura_written:
        return pleura["vector"], pleura["state"]

    return lung["vector"], lung["state"]


def load_text(root, split, regions):
    """
    Read the sentence table and text vectors, and build text + state for every scan.

    The table has one row per scan and region (10 regions, incl. pleura), and row i
    matches row i of the vector file. For each scan we:
      - merge pleura into lung
      - put the 9 regions in the order the model uses
      - leave SILENT regions as zeros (their vector comes from an empty sentence)

    Args:
        root: data folder (sample_data locally, silence_ct on Colab)
        split: "valid" or "train"
        regions: list of the 9 region names, in the order the model uses

    Returns:
        dict {scan: (text (9, 768), state (9,))}, state: 1 ABNORMAL, 0 NORMAL, -1 SILENT
    """

    table = pd.read_csv(f"{root}/text_vectors/sentences_{split}.csv")
    vectors = np.load(f"{root}/text_vectors/text_{split}.npy").astype(np.float32)

    # group rows by scan: {scan: {region: {"vector": ..., "state": ...}}}
    rows_by_scan = {}
    for i, row in enumerate(table.itertuples(index=False)):
        scan = row.Volumename.replace(".nii.gz", "")
        if scan not in rows_by_scan:
            rows_by_scan[scan] = {}
        rows_by_scan[scan][row.region] = {"vector": vectors[i], "state": row.state}

    text_by_scan = {}
    for scan, rows in rows_by_scan.items():
        text = np.zeros((len(regions), vectors.shape[1]), dtype=np.float32)
        state = np.full(len(regions), STATE_IDS["SILENT"], dtype=np.int64)

        lung_vector, lung_state = merge_lung_and_pleura(rows["lung"], rows["pleura"])
        rows["lung"] = {"vector": lung_vector, "state": lung_state}

        for k, region in enumerate(regions):
            if rows[region]["state"] == "SILENT":
                continue                      # keep zeros and -1
            text[k] = rows[region]["vector"]
            state[k] = STATE_IDS[rows[region]["state"]]

        text_by_scan[scan] = (text, state)

    return text_by_scan


def load_geometry(path, regions):
    """
    Read one scan's geometry file and return distance and touching between regions.

    The geometry file stores regions in alphabetical order, so we reorder rows and
    columns to match the order the model uses. If a region mask was empty, its center
    is NaN, so its distances are NaN; we replace them with a large distance (far away).

    Args:
        path: geometry file of one scan, e.g. geometry/valid/valid_1001_a_1.npz
        regions: list of the 9 region names, in the order the model uses

    Returns:
        distance: (9, 9) distance between region centers, in mm
        touching: (9, 9) 1 if two regions share a boundary, else 0
    """

    g = np.load(path)
    saved_regions = list(g["regions"])

    # position of each model region inside the saved file
    order = []
    for region in regions:
        order.append(saved_regions.index(region))

    # reorder rows, then columns
    distance = g["distance"][order][:, order].astype(np.float32)
    touching = g["touching"][order][:, order].astype(np.float32)

    distance[np.isnan(distance)] = FAR_DISTANCE_MM
    return distance, touching

class SilenceDataset(Dataset):
    """
    One item = one CT scan, with all inputs for its 9 regions.

    Each time a scan is loaded, some WRITTEN regions are hidden at random. The model
    sees their image and the rest of the report, and must predict their text vector.
    SILENT regions are never hidden: they have no true text, so there is no answer.

    Args:
        root: data folder (sample_data locally, silence_ct on Colab)
        split: "valid" or "train"
        hide_prob: chance that each written region is hidden
        seed: random seed, so the hidden regions can be repeated
    """
    # TODO: partial silence (hide some sentences) and finding silence (hide one finding)

    def __init__(self, root, split, hide_prob=0.3, seed=0):
        self.root = root
        self.split = split
        self.hide_prob = hide_prob
        self.rng = np.random.default_rng(seed)
        self.regions = list(REGION_TO_ORGANS)

        self.text = load_text(root, split, self.regions)

        # keep only scans that have organ vectors, geometry and text
        organ_scans = {Path(f).stem for f in glob.glob(f"{root}/organ_vectors/{split}/*.npy")}
        geo_scans = {Path(f).stem for f in glob.glob(f"{root}/geometry/{split}/*.npz")}
        self.scans = sorted(organ_scans & geo_scans & set(self.text))

    def __len__(self):
        return len(self.scans)

    def choose_hidden(self, state):
        """
        Pick which written regions to hide.

        Each written region is hidden with probability hide_prob. If none is picked,
        we hide one at random, so every scan has at least one answer to learn from.

        Returns:
            is_hidden: (9,) True for hidden regions
        """
         
        is_hidden = np.zeros(len(state), dtype=bool)

        written = []
        for k in range(len(state)):
            if state[k] != STATE_IDS["SILENT"]:
                written.append(k)

        if not written:
            return is_hidden               # whole report silent: nothing to hide

        for k in written:
            if self.rng.random() < self.hide_prob:
                is_hidden[k] = True

        if not is_hidden.any():
            is_hidden[self.rng.choice(written)] = True
        return is_hidden

    def __getitem__(self, idx):
        """
        Load one scan and hide some regions.

        Returns a dict of tensors:
            image (9, 1024), has_image (9,)    OrganLens vectors
            text (9, 768), has_text (9,)       text the model sees (zeros if silent or hidden)
            target (9, 768), is_hidden (9,)    true text of hidden regions (the answer)
            state (9,)                         1 ABNORMAL, 0 NORMAL, -1 SILENT
            distance (9, 9), touching (9, 9)   geometry

        # inputs (x): image, has_image, text, has_text, distance, touching
        # output (y): target, only where is_hidden is True
        """

        scan = self.scans[idx]

        organ_vectors = np.load(f"{self.root}/organ_vectors/{self.split}/{scan}.npy").astype(np.float32)
        image, has_image = organs_to_regions(organ_vectors, self.regions)

        text, state = self.text[scan]
        is_hidden = self.choose_hidden(state)

        # the answer: true text of hidden regions only
        target = text.copy()
        target[~is_hidden] = 0

        # what the model sees: hidden regions become zeros
        visible_text = text.copy()
        visible_text[is_hidden] = 0
        has_text = (state != STATE_IDS["SILENT"]) & ~is_hidden

        distance, touching = load_geometry(f"{self.root}/geometry/{self.split}/{scan}.npz", self.regions)

        return {
            "scan": scan,
            "image": torch.from_numpy(image),
            "has_image": torch.from_numpy(has_image),
            "text": torch.from_numpy(visible_text),
            "has_text": torch.from_numpy(has_text),
            "target": torch.from_numpy(target),
            "is_hidden": torch.from_numpy(is_hidden),
            "state": torch.from_numpy(state),
            "distance": torch.from_numpy(distance),
            "touching": torch.from_numpy(touching),
        }


if __name__ == "__main__":
    # check on local sample: 
    # PYTHONPATH=src python -m silence_ct.data.dataset
    
    root = "/content/drive/MyDrive/silence_ct"
    if not Path(root).exists():
        root = "data"

    for split in ["train", "valid"]:
        if not Path(f"{root}/geometry/{split}").exists():
            print(split, "not found, skip")
            continue

        dataset = SilenceDataset(root, split)
        print(f"\n{split}: {len(dataset)} scans")

        item = dataset[0]
        print("scan:", item["scan"])
        for key, value in item.items():
            if key != "scan":
                print(f"  {key:10s} {tuple(value.shape)}")

        # NaN check on the first 200 scans
        nan_count = 0
        for index in range(min(200, len(dataset))):
            item = dataset[index]
            for key in ["image", "text", "distance"]:
                if torch.isnan(item[key]).any():
                    nan_count += 1
                    break
        print("  scans with NaN (first 200):", nan_count)
