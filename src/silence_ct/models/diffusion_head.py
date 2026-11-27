"""
Prediction heads

Turn each encoded region token into a predicted text vector (CT-CLIP, 768).
Flow: encoder.py -> head -> predicted text vector.

    RegressionHead: one predicted vector per region (baseline)
    DiffusionHead:  several possible vectors per region (TODO)

Input:  encoded tokens (B, 9, d_model)
Output: predicted text vectors (B, 9, 768)
"""

import torch
from torch import nn


class RegressionHead(nn.Module):
    """
    Predict one text vector for every region with a small MLP.

    It predicts for all 9 regions, but which predictions are used depends on the step:
      - training -> only hidden regions go into the loss
      - test     -> hidden regions are scored; silent regions are the ones we fill

    Args:
        d_model: size of each encoded token
        text_dim: size of the CT-CLIP text vector (768)
        dropout: dropout rate
    """

    def __init__(self, d_model=256, text_dim=768, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, text_dim),
        )

    def forward(self, encoded):
        """
        Args:
            encoded: (B, 9, d_model) tokens from GeometryEncoder

        Returns:
            predicted: (B, 9, 768) predicted text vector per region
        """
        return self.mlp(encoded)


# TODO: DiffusionHead (noise prediction conditioned on the encoded token)


if __name__ == "__main__":
    # PYTHONPATH=src python -m silence_ct.models.diffusion_head
    encoded = torch.randn(2, 9, 256)
    predicted = RegressionHead()(encoded)
    print("predicted:", tuple(predicted.shape))     #  (2, 9, 768)