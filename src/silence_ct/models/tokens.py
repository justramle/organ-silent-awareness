"""
Region tokens (model input)

Turns the inputs from SilenceDataset into tokens that the
encoder reads

Every region turn to one token w three parts:
    - image part:  OrganLens vector (1024) projected to d_model, or a learned
                   "no image" vector (bone, breast, thyroid)
    - text part:   CT-CLIP vector (768) projected to d_model, or a learned
                   "no text" vector (region is silent or hidden)
    - region part: learned region embedding, so the model knows which region it is

Input:  image (B, 9, 1024), has_image (B, 9), text (B, 9, 768), has_text (B, 9)
Output: tokens (B, 9, d_model), one token per region
"""

import torch
from torch import nn


class RegionTokens(nn.Module):
    """
    Turn the image and text vectors of 9 regions into 9 tokens of size d_model.

    For each region:
      - has image -> project image (1024 -> d_model), else learned no_image vector
      - has text  -> project text (768 -> d_model), else learned no_text vector
                     (silent and hidden regions both have no text)
      - always add the region embedding, so the model knows which region it is

    Args:
        num_regions: number of regions (9)
        image_dim: size of the image vector (1024)
        text_dim: size of the text vector (768)
        d_model: size of each token
    """

    def __init__(self, num_regions=9, image_dim=1024, text_dim=768, d_model=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)         #nn.Linear(1024, 256) learnable w
        self.text_proj = nn.Linear(text_dim, d_model)           # nn.Linear(768, 256)
        self.region_embed = nn.Embedding(num_regions, d_model)  # nn.Embedding(9, 256) 9 learnable vec

        self.no_image = nn.Parameter(torch.zeros(d_model))      # 1 learn vec
        self.no_text = nn.Parameter(torch.zeros(d_model))       # 1
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image, has_image, text, has_text):
        """
        Build the tokens for a batch of scans.

        Args:
            image: (B, 9, 1024) image vectors
            has_image: (B, 9) True if the region has an image vector
            text: (B, 9, 768) text vectors (zeros where silent or hidden)
            has_text: (B, 9) True if the region has visible text

        Returns:
            tokens: (B, 9, d_model)
        """
        image_part = self.image_proj(image)
        image_part = torch.where(has_image.unsqueeze(-1), image_part, self.no_image)

        text_part = self.text_proj(text)
        text_part = torch.where(has_text.unsqueeze(-1), text_part, self.no_text)

        # region ids 0..8, same for every scan in the batch
        region_ids = torch.arange(image.shape[1], device=image.device)
        region_part = self.region_embed(region_ids)

        tokens = image_part + text_part + region_part
        return self.norm(tokens)


if __name__ == "__main__":
    #  PYTHONPATH=src python -m silence_ct.models.tokens
    batch_size = 4
    image = torch.randn(batch_size, 9, 1024)
    has_image = torch.rand(batch_size, 9) > 0.3
    text = torch.randn(batch_size, 9, 768)
    has_text = torch.rand(batch_size, 9) > 0.5

    model = RegionTokens()
    tokens = model(image, has_image, text, has_text)
    print("tokens:", tuple(tokens.shape))     #  (4, 9, 256)
