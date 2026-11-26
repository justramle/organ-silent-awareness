"""
Full model: tokens -> geometry-biased encoder -> head.

    1. tokens.py         region tokens from image vectors, text vectors, region embeddings
    2. encoder.py        transformer with geometry bias, so regions share information
    3. diffusion_head.py predict the sentence vector of each hidden region
    4. retrieval.py      (inference only) nearest real sentence

The head type ("diffusion" or "regression") and the bias type ("none", "touch", "dist",
"both") are set in the config, so all baselines and ablations use the same code.
"""
