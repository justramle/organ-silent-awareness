"""
Prediction heads: from the encoder token of a hidden region to its sentence vector (768-d).

DiffusionHead (main)
    A small denoising network conditioned on the region token. During training, noise is
    added to the true sentence vector and the head learns to remove it. At test time we
    start from noise and denoise step by step. Sampling several times gives several
    possible sentences; their spread is used as an uncertainty estimate.

RegressionHead (baseline)
    A small MLP that predicts the sentence vector directly in one step. Same input as the
    diffusion head, but only one answer and no uncertainty.

Comparing the two heads tests whether modeling several possible sentences helps.
"""
