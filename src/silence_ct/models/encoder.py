"""
Geometry-aware region encoder

A small transformer over the 9 region tokens. Each region looks at the other regions
(self-attention), with the geometry bias added to the attention scores, so nearby
regions matter more. Flow: tokens.py -> encoder.py -> head.

Input:  tokens (B, 9, d_model), distance (B, 9, 9), touching (B, 9, 9)
Output: encoded tokens (B, 9, d_model)
"""

import torch
from torch import nn

from silence_ct.models.geometry_bias import GeometryBias


class GeometryEncoder(nn.Module):
    """
    Transformer encoder with a geometry bias in every attention layer.

    The same bias is used in all layers:
      - mode "both" / "dist" / "touch" -> bias added to attention scores
      - mode "none"                    -> plain transformer (content only, ablation A)

    Args:
        d_model: size of each token
        num_heads: number of attention heads
        num_layers: number of transformer layers
        dropout: dropout rate
        mode: geometry mode for GeometryBias
    """

    def __init__(self, d_model=256, num_heads=4, num_layers=3, dropout=0.1, mode="both"):
        super().__init__()
        self.num_heads = num_heads
        self.geometry_bias = GeometryBias(num_heads, mode)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model, num_heads, 
                dim_feedforward=4 * d_model,
                dropout=dropout, 
                batch_first=True, 
                norm_first=True,
            )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, distance, touching):
        """
        Let regions look at each other, guided by geometry.

        Args:
            tokens: (B, 9, d_model) region tokens from RegionTokens
            distance: (B, 9, 9) distance between regions, in mm
            touching: (B, 9, 9) 1 if two regions touch, else 0

        Returns:
            encoded: (B, 9, d_model)
        """
        bias = self.geometry_bias(distance, touching)

        # PyTorch attention takes the bias as a float mask of shape (B * heads, 9, 9)
        attention_mask = None
        if bias is not None:
            batch_size, num_heads, num_regions, _ = bias.shape
            attention_mask = bias.reshape(batch_size * num_heads, num_regions, num_regions)

        encoded = tokens
        for layer in self.layers:
            encoded = layer(encoded, src_mask=attention_mask)
        return self.norm(encoded)


if __name__ == "__main__":
    # PYTHONPATH=src python -m silence_ct.models.encoder
    tokens = torch.randn(4, 9, 256)
    distance = torch.rand(4, 9, 9) * 300
    touching = (torch.rand(4, 9, 9) > 0.7).float()

    for mode in ["none", "both"]:
        encoder = GeometryEncoder(mode=mode)
        encoded = encoder(tokens, distance, touching)
        print(mode, "->", tuple(encoded.shape))     # (4, 9, 256)