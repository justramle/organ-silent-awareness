# Model architecture

Final model: geometry = none (the geometry ablation showed no gain), state head + diffusion head.
The geometry bias is kept as an option and drawn below for completeness.

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
┌──────────────────────────────────────────────────────────────────┐
│ 3. HEADS                                                         │
│  state head (model.py):         256 → 1    P(abnormal)           │
│  RegressionHead (stage 1):      256 → 256 → 768   one vector     │
│  DiffusionHead (stage 2):       small MLP denoiser, predicts the │
│                                 clean vector; 10 samples/region  │
└──────────────────┬───────────────────────────────────────────────┘
                   │ P(abnormal) (64, 9) + text vector(s) (64, 9, 768)
                   ▼
     ┌─────────────┴──────────────┐
     │ TRAINING                   │ TEST
     ▼                            ▼
 stage 1: (1 − cosine) +          state head → P(abnormal)
   binary cross-entropy           vector(s) → retrieval.py: closest real
 stage 2: clean-vector MSE          sentence in the train bank
 ONLY where is_hidden             diffusion: spread of samples = uncertainty
```
