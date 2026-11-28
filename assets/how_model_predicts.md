# How the model predicts one silent region

Example: the esophagus is SILENT in the report.

```
                     ┌─────────────────────────────────────────────┐
  esophagus image ──▶│ 1. OWN image: what the esophagus looks like │  ← strongest evidence
                     │                                             │
  heart, trachea,    │ 2. NEIGHBORS: other regions' image + text   │  ← supporting evidence
  lung… (image+text)─▶│    "heart is enlarged", "trachea normal"…   │
                     │                                             │
  geometry ─────────▶│ 3. GEOMETRY decides HOW MUCH to listen      │  ← not evidence itself,
                     │    to each neighbor (near = more)           │     a "map"
                     └──────────────────────┬──────────────────────┘
                                            ▼
                         "Esophagus is dilated." (ABNORMAL)
```

---

## Key words

- **Silent region**: the report says nothing about it. The model fills **every** silent region (all 9 regions).
- **Own image**: the region's OrganLens vector. The main evidence, like a doctor looking at the organ itself.
- **Neighbor**: any other region, especially nearby ones. Their image and text give hints.
- **Geometry**: not something to fill. It is the map (distance, touching) that tells the model which neighbors to trust more.

---

## Regions without an OrganLens image

OrganLens has no vector for **thyroid, breast, bone**. For these, step 1 is missing and the model uses only
neighbors + geometry + the rest of the report (a learned "no image" vector takes the place of the image).
Later: pool our own image vector inside the RadGenome mask (option B) and compare.

---

## Main organs = where we measure, not what we fill

To check if a prediction is right we need the true answer, so we test on **written** regions:
hide the sentence, predict it, compare. Heart and esophagus have many written examples (abnormal and normal),
so their scores are the most reliable. The model still fills all silent regions.

---

## Update (Nov 27): geometry in the final model

The geometry ablation (3 seeds) showed that distance / touching give **no reliable gain** at the 9-region level,
so the final model uses **geometry = none**: the region embedding already tells the model which organs are
neighbours, because the layout is almost the same in every patient. Step 3 above is kept as an option.
The current full pipeline (state head + diffusion head) is in `full_pipeline.md`.
