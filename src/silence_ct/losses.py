"""
Training losses.

Text-deletion loss
    Computed only on regions whose sentences were hidden by the dataset. The true sentence
    vector is the target.
        regression head: cosine distance between predicted and true sentence vector
        diffusion head:  mean squared error between predicted and true noise

Silence-aware rule
    Regions that are SILENT in the original report have no target and get no loss.
    Silence means "unknown", not "normal"; most existing labels treat it as normal.
"""
