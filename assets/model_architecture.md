# Model architecture

Shapes shown for a batch of 64 scans (9 regions each).

```
════════════════════════════════════════════════════════════════════════
 INPUT (from dataset.py)                         one scan = 9 regions
════════════════════════════════════════════════════════════════════════
 image (64,9,1024)  has_image (64,9)        distance (64,9,9)
 text  (64,9,768)   has_text  (64,9)        touching (64,9,9)
        │                                          │
        ▼                                          │
┌──────────────────────────────┐                   │
│ 1. TOKENS  (tokens.py)       │                   │
│  image ─Linear─► 256 ┐       │                   │
│  text  ─Linear─► 256 ├─ +    │                   │
│  region embedding    ┘       │                   │
└──────────────┬───────────────┘                   │
               │ tokens (64, 9, 256)               │
               ▼                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ 2. ENCODER  (encoder.py) = 3 transformer layers stacked           │
│                                                                   │
│        ┌──────────────── geometry_bias.py ◄──────────────┐        │
│        │  bias = -w·distance + b·touching   (64,4,9,9)   │        │
│        ▼  (same bias given to every layer)               │        │
│  ┌─────────────────────────────────────────────┐                  │
│  │ LAYER 1                                     │                  │
│  │  ┌─────────────────────────────────────┐    │                  │
│  │  │ Self-attention, 4 heads             │    │                  │
│  │  │ each region looks at all 9 regions  │    │                  │
│  │  │ score(i,j) = q_i·k_j/√d + bias(i,j) │    │                  │
│  │  │ softmax → mix the regions' info     │    │                  │
│  │  └─────────────────────────────────────┘    │                  │
│  │        + (residual: add input back)         │                  │
│  │  ┌─────────────────────────────────────┐    │                  │
│  │  │ Feed-forward: 256 → 1024 → 256      │    │                  │
│  │  │ (each region thinks on its own)     │    │                  │
│  │  └─────────────────────────────────────┘    │                  │
│  │        + (residual)                         │                  │
│  └──────────────────┬──────────────────────────┘                  │
│                     ▼  (64, 9, 256)                               │
│  ┌─────────── LAYER 2 (same structure) ───────┐                   │
│  └──────────────────┬──────────────────────────┘                  │
│                     ▼                                             │
│  ┌─────────── LAYER 3 (same structure) ───────┐                   │
│  └──────────────────┬──────────────────────────┘                  │
└─────────────────────┼─────────────────────────────────────────────┘
                      │ encoded (64, 9, 256)
                      │ each token now knows about its neighbours
                      ▼
┌──────────────────────────────────────┐
│ 3. HEAD  (diffusion_head.py)         │
│  RegressionHead: 256 → 256 → 768     │
│  (same small MLP for each region)    │
└──────────────────┬───────────────────┘
                   │ predicted (64, 9, 768)
                   ▼
     ┌─────────────┴──────────────┐
     │ TRAINING                   │ TEST
     ▼                            ▼
 loss = 1 - cosine(pred, target)  retrieval.py: closest real
 ONLY where is_hidden             sentence in the train bank
 → update all weights             → output text
```
