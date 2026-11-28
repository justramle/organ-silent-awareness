# Silence-Aware CT

Silence-aware CT is a multimodal learning method for missing organ-level information in 3D CT.

When a CT report says nothing about an organ, we do **not** assume it is normal. We treat the organ as
**unknown** and predict, from its 3D image and the rest of the report:

- **P(abnormal)** for the silent organ (state head)
- **possible sentences** a radiologist could have written (diffusion head + retrieval of real sentences)
- **how sure** the model is (spread of the diffusion samples)

---

## Three kinds of silence

| Level | Meaning | Status |
|---|---|---|
| Region (full) | nothing is written about the region | implemented |
| Sub-region (partial) | some parts of the region are written, others are not | later |
| Finding (implicit) | the region is written, but a finding is left out | later |

---

## Model


```
organ vectors (OrganLens) ─┐                                ┌─ state head     → P(abnormal)
text vectors (CT-CLIP)    ─┼─> 9 region tokens ─> encoder ──┤
                           │      (3 transformer layers)    └─ diffusion head → 10 text vectors
                           │                                     (stage 2)        → nearest real sentences
geometry (optional) ───────┘                                                      → spread = uncertainty
```

- Frozen foundation models give the inputs: OrganLens (image, per organ) and CT-CLIP (text).
- Stage 1 (`train.py`): tokens → encoder → state head + regression head. Trained by hiding written regions.
- Stage 2 (`train_diffusion.py`): diffusion head on top of the frozen stage-1 model.
- Geometry bias (distance / touching between organs) is an option; the ablation showed no gain, so the final
  model uses `geometry: none`.
- Ideas: (BERT masking, FT-Transformer, Graphormer, MAR, DALL·E 2 prior,
  CXR-RePaiR retrieval).

---

## Results (test = RadGenome valid split, 1,564 scans, leave-one-region-out, 10,398 written regions)

| Method | Heart AUC | Esophagus AUC | Macro AUC (9 regions) |
|---|---|---|---|
| silence = normal (baseline) | 0.500 | 0.500 | 0.500 |
| regression head → sentence | 0.759 | 0.543 | 0.633 |
| diffusion head → sentences | 0.812 | 0.763 | 0.723 |
| **state head** | **0.862** | **0.791** | **0.788** |

Geometry ablation (3 seeds, macro AUC): none 0.787 ± 0.003, both 0.770 ± 0.004 → geometry not used.
Likely reason: at the level of 9 large regions the layout is almost the same in every patient, so the region
embedding already encodes which organs are neighbours. Geometry may still help at a finer level
(sub-regions such as lung lobes, or lesion locations) — future work.

---

## Data

Data is not stored in this repo. It is prepared in Colab and saved in Google Drive (`silence_ct/`):

- `organ_vectors/<split>/`: one (11, 1024) vector file per scan (OrganLens)
- `text_vectors/`: sentence table and (n, 768) sentence vectors (CT-CLIP)
- `radgenome/state_table_<split>.csv`: ABNORMAL / NORMAL / SILENT per region
- `geometry/<split>/`: center, volume, touching, distance per scan
- `cache/image_geometry_<split>.npz`: one file per split, built on the first run (fast loading)
- `runs/<name>/`: trained models, logs, evaluation tables and figures

Splits: 5,000 RadGenome train scans → 4,500 train / 500 validation (stratified by heart and esophagus state);
test = 1,564 RadGenome validation scans.

Later: **NLST** as a zero-shot transfer evaluation (silent at year 0 → finding at a later year?).

---

## Layout

```
silence_aware_ct/
├── README.md
├── requirements.txt
├── assets/                       # diagrams and short explanations (full_pipeline.md, model_flow.md, ...)
├── configs/
│   └── config.yaml               # all settings (stage 1 and diffusion)
├── data/                         # small local copy for quick tests (not in git)
└── src/silence_ct/
    ├── data/
    │   └── dataset.py            # load vectors, states, geometry (cached); hide written regions
    ├── models/
    │   ├── tokens.py             # region tokens from image and text vectors
    │   ├── geometry_bias.py      # optional -w·distance + b·touching attention bias (zero-init)
    │   ├── encoder.py            # transformer layers
    │   ├── diffusion_head.py     # RegressionHead + DiffusionHead
    │   ├── retrieval.py          # nearest real train sentence
    │   └── model.py              # tokens -> encoder -> state head + regression head
    ├── losses.py                 # cosine loss + state loss (hidden regions only)
    ├── train.py                  # stage 1 training
    ├── train_diffusion.py        # stage 2: diffusion head on the frozen stage-1 model
    ├── evaluate.py               # leave-one-region-out test, baselines, bootstrap CI, ROC figures
    └── plot.py                   # training curves
```

---

## Run

```bash
pip install -r requirements.txt

# stage 1
PYTHONPATH=src python -m silence_ct.train --config configs/config.yaml
# stage 2
PYTHONPATH=src python -m silence_ct.train_diffusion --base_run <data_root>/runs/regression_none_state
# test + figures
PYTHONPATH=src python -m silence_ct.evaluate --run_dir <data_root>/runs/diffusion_none_state
PYTHONPATH=src python -m silence_ct.plot --run_dir <data_root>/runs/diffusion_none_state
```
