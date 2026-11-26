# Silence-Aware CT

Silence-aware CT is a multimodal learning method for missing organ-level information in 3D CT.

When a CT report says nothing about an organ, we predict the missing sentence for that organ
from its 3D image, its neighboring organs, and the rest of the report.

---

## Three kinds of silence

| Level | Meaning |
|---|---|
| Region (full) | nothing is written about the region |
| Sub-region (partial) | some parts of the region are written, others are not |
| Finding (implicit) | the region is written, but a finding is left out |

The model is the same for all three. Only the sentences we hide during training change.

---

## Model

```
organ vectors (OrganLens) ─┐
text vectors (CT-CLIP)    ─┼─> tokens ─> transformer with geometry bias ─> head ─> sentence vector ─> nearest real sentence
geometry (masks)          ─┘                (-w·distance + b·touching)      (diffusion or regression)
```

---

## Data

Data is not stored in this repo. It is prepared in Colab and saved in Google Drive (`silence_ct/`):

- `organ_vectors/<split>/`: one (11, 1024) vector file per scan (OrganLens)
- `text_vectors/`: sentence table and (n, 768) sentence vectors (CT-CLIP)
- `radgenome/state_table_<split>.csv`: ABNORMAL / NORMAL / SILENT per region
- `geometry/geometry_<split>.npz`: center, volume, touching, distance per region

Splits: train = 5,000 RadGenome train scans, test = 1,564 RadGenome validation scans (used once).

Later: **NLST** (lung screening CT with yearly follow-up). Used as an external check over time:
if the model predicts a finding for a silent organ at year 0, does that finding appear at a later year?

---

## Layout

```
silence_aware_ct/
├── README.md
├── requirements.txt
├── assets/                       # figures and short explanations (e.g. how_model_predicts.md)
├── configs/
│   └── config.yaml               # paths and settings
├── notebooks/                    # Colab runner
├── sample_data/                  # small local sample for testing (not in git)
└── src/silence_ct/
    ├── data/
    │   └── dataset.py            # load vectors, states, geometry; choose what to hide
    ├── models/
    │   ├── tokens.py             # region tokens from image and text vectors
    │   ├── geometry_bias.py      # -w·distance + b·touching attention bias
    │   ├── encoder.py            # transformer blocks that use the bias
    │   ├── diffusion_head.py     # DiffusionHead (main) + RegressionHead (baseline)
    │   ├── retrieval.py          # nearest real sentence
    │   └── model.py              # tokens -> encoder -> head
    ├── losses.py                 # training losses
    ├── train.py                  # training loop
    └── evaluate.py               # 5 levels, including the flip test
```

---

## Setup

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m silence_ct.train --config configs/config.yaml
```
