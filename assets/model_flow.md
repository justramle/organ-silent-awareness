# Model flow

How the inputs (x) from `dataset.py` become a predicted text vector.

```
dataset.py gives x:
  image, has_image, text, has_text      distance, touching
              │                                  │
              ▼                                  │
        tokens.py                                │
   9 tokens (one per region)                     │
              │                                  ▼
              └──────────────►  encoder.py  ◄── geometry_bias.py
                                (regions look at each other,
                                 nearby regions matter more)
                                     │
                                     ▼
                            diffusion_head.py
                          predicted text vector
                                     │
                                     ▼
                compare with target (y), only where is_hidden
```

- **x**: raw vectors from the dataset (image, text, geometry).
- **tokens**: one vector of size 256 per region (image part + text part + region part).
- **y (target)**: true text vector of hidden regions, used only in the loss.
