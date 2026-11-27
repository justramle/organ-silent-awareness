"""
Silence-aware region model

The full model: tokens -> geometry-aware encoder -> head. Takes one batch from
SilenceDataset and predicts a text vector for every region (see assets/model_flow.md).

Input:  batch dict with image, has_image, text, has_text, distance, touching
Output: predicted text vectors (B, 9, 768) and abnormal logits (B, 9)
"""

import torch
from torch import nn

from silence_ct.models.tokens import RegionTokens
from silence_ct.models.encoder import GeometryEncoder
from silence_ct.models.diffusion_head import RegressionHead


class SilenceModel(nn.Module):
    """
    Predict the text vector and the normal/abnormal state of each region from its
    image, the other regions' text and the geometry between regions.

      - tokens:     image + text + region embedding -> one token per region
      - encoder:    regions look at each other, nearby regions matter more
      - head:       token -> predicted text vector (sentence via retrieval)
      - state head: token -> logit of "abnormal" (multi-task, shares the encoder)

    Args:
        d_model: size of each token
        num_heads: number of attention heads
        num_layers: number of transformer layers
        dropout: dropout rate
        mode: geometry mode ("none", "dist", "touch", "both")
    """

    def __init__(self, d_model=256, num_heads=4, num_layers=3, dropout=0.1, mode="both"):
        super().__init__()
        self.tokens = RegionTokens(d_model=d_model)
        self.encoder = GeometryEncoder(d_model, num_heads, num_layers, dropout, mode)
        self.head = RegressionHead(d_model, dropout=dropout)
        self.state_head = nn.Linear(d_model, 1)

    def forward(self, batch):
        """
        Args:
            batch: dict from SilenceDataset (batched by DataLoader), with
                image (B, 9, 1024), has_image (B, 9), text (B, 9, 768),
                has_text (B, 9), distance (B, 9, 9), touching (B, 9, 9)

        Returns:
            dict with vector (B, 9, 768) and abnormal_logit (B, 9)
        """
        tokens = self.tokens(batch["image"], batch["has_image"],
                             batch["text"], batch["has_text"])
        encoded = self.encoder(tokens, batch["distance"], batch["touching"])
        return {
            "vector": self.head(encoded),
            "abnormal_logit": self.state_head(encoded).squeeze(-1),
        }


if __name__ == "__main__":
    # PYTHONPATH=src python -m silence_ct.models.model
    batch_size = 2
    batch = {
        "image": torch.randn(batch_size, 9, 1024),
        "has_image": torch.rand(batch_size, 9) > 0.3,
        "text": torch.randn(batch_size, 9, 768),
        "has_text": torch.rand(batch_size, 9) > 0.5,
        "distance": torch.rand(batch_size, 9, 9) * 300,
        "touching": (torch.rand(batch_size, 9, 9) > 0.7).float(),
    }

    model = SilenceModel()
    outputs = model(batch)
    num_weights = sum(p.numel() for p in model.parameters())
    print("vector:", tuple(outputs["vector"].shape))                  #  (2, 9, 768)
    print("abnormal_logit:", tuple(outputs["abnormal_logit"].shape))  #  (2, 9)
    print("weights:", num_weights)