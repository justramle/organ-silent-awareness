# Model flow

How the inputs (x) from `dataset.py` become the model outputs.

```
dataset.py gives x:
  image, has_image, text, has_text      distance, touching
              │                                  │
              ▼                                  │
        tokens.py                                │
   9 tokens (one per region)                     │
              │                                  ▼
              └──────────────►  encoder.py  ◄── geometry_bias.py
                                (regions look at each other)   (optional; final model: geometry = none)
                                     │
                     ┌───────────────┼──────────────────────┐
                     ▼               ▼                      ▼
              state head      regression head        diffusion head (stage 2)
              P(abnormal)     one text vector        several text vectors
                     │               │                      │
                     │               └──────────┬───────────┘
                     │                          ▼
                     │                 retrieval.py → real train sentence
                     ▼
        loss (training): only where is_hidden
          state: binary cross-entropy   vector: 1 − cosine   diffusion: clean-vector MSE
```

- **x**: raw vectors from the dataset (image, text, geometry).
- **tokens**: one vector of size 256 per region (image part + text part + region part).
- **y (target)**: true text vector and true state of hidden regions, used only in the losses.
- Full example with one silent region: `full_pipeline.md`.
