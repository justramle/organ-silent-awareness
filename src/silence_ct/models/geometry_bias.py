"""
Geometry attention bias

Turns region geometry into a bias added to attention scores, so regions that are
close or touching look at each other more

    bias(i, j) = - w * distance(i, j) + b * touching(i, j)

w and b are learned, one pair per attention head. The mode setting turns each
signal on or off for the ablation (H2): "none", "dist", "touch", "both".

Input:  distance (B, 9, 9) in mm, touching (B, 9, 9) 0/1
Output: bias (B, num_heads, 9, 9)
"""

import torch
from torch import nn

# distance is in mm (0-300); divide so the numbers are around 0-3
DISTANCE_SCALE_MM = 100.0


class GeometryBias(nn.Module):
    """
    Compute one attention bias matrix per head from distance and touching.

    Each head learns its own weights:
      - w (distance): far regions get a lower score
      - b (touching): touching regions get a higher score
      - mode turns signals off: "none" -> no bias, "dist" / "touch" -> one signal

    Args:
        num_heads: number of attention heads
        mode: "none", "dist", "touch" or "both"
    """

    def __init__(self, num_heads=4, mode="both"):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.distance_weight = nn.Parameter(torch.ones(num_heads))
        self.touch_weight = nn.Parameter(torch.ones(num_heads))

    def forward(self, distance, touching):
        """
        Build the bias for a batch of scans.

        Args:
            distance: (B, 9, 9) distance between region centers, in mm
            touching: (B, 9, 9) 1 if two regions touch, else 0

        Returns:
            bias: (B, num_heads, 9, 9), or None when mode is "none"
        """
        if self.mode == "none":
            return None

        batch_size, num_regions, _ = distance.shape
        bias = torch.zeros(batch_size, self.num_heads, 
                           num_regions, num_regions,
                           device=distance.device)

        # (num_heads,) -> (1, num_heads, 1, 1) so it multiplies each head
        w = self.distance_weight.view(1, -1, 1, 1)
        b = self.touch_weight.view(1, -1, 1, 1)

        if self.mode in ("dist", "both"):
            scaled_distance = distance.unsqueeze(1) / DISTANCE_SCALE_MM
            bias = bias - w * scaled_distance

        if self.mode in ("touch", "both"):
            bias = bias + b * touching.unsqueeze(1)

        return bias


if __name__ == "__main__":
    #  PYTHONPATH=src python -m silence_ct.models.geometry_bias
    batch_sz = 2 #2 scans
    distance = torch.rand(batch_sz, 9, 9) * 300
    touching = (torch.rand(batch_sz, 9, 9) > 0.7).float()

    for mode in ["none", "dist", "touch", "both"]:
        bias = GeometryBias(num_heads=4, mode=mode)(distance, touching)
        if bias is None:
            print(mode, "-> None")
        else:
            print(mode, "->", tuple(bias.shape))     #  (2: 2scan in B, 4: Head, 9: region row, 9: region col)