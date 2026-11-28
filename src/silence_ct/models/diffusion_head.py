"""
Prediction heads

Turn each encoded region token into a predicted text vector (CT-CLIP, 768).
Flow: encoder.py -> head -> predicted text vector.

    RegressionHead: one predicted vector per region (baseline)
    DiffusionHead:  a distribution over vectors per region; sampling gives several
                    possible vectors (MAR-style small MLP denoiser, Li et al. 2024)

Input:  encoded tokens (B, 9, d_model) or (N, d_model)
Output: predicted text vectors (B, 9, 768), or samples (N, num_samples, 768)
"""

import math

import torch
import torch.nn.functional as F
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


def cosine_schedule(num_steps, offset=0.008):
    """
    Share of signal left at each noise step (cosine schedule, Nichol & Dhariwal 2021).

      - step 0    -> almost all signal (alpha_bar close to 1)
      - last step -> almost pure noise (alpha_bar close to 0)

    Args:
        num_steps: number of noise steps
        offset: small shift so the first step is not exactly noise-free

    Returns:
        alpha_bar: (num_steps,)
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    curve = torch.cos((steps + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = curve / curve[0]
    return alpha_bar[1:].clamp(min=1e-5, max=0.9999).float()


class DiffusionHead(nn.Module):
    """
    Model the distribution of a region's text vector with a small denoising MLP.

      - training: add noise to the true vector at a random step; the MLP predicts the
                  noise from the noisy vector, the step and the region's token (condition)
      - sampling: start from random noise and remove it step by step (DDIM);
                  different start noise -> different samples
    Text vectors are compared by direction, so targets are scaled to length sqrt(768)
    (about unit size per number) and samples are returned with length 1.

    Args:
        d_model: size of the encoded token (condition)
        text_dim: size of the CT-CLIP text vector (768)
        width: hidden size of the MLP
        num_blocks: number of residual blocks
        num_steps: number of noise steps in training
    """

    def __init__(self, d_model=256, text_dim=768, width=512, num_blocks=3, num_steps=1000):
        super().__init__()
        self.text_dim = text_dim
        self.width = width
        self.num_steps = num_steps
        self.radius = math.sqrt(text_dim)

        self.input_proj = nn.Linear(text_dim, width)
        self.condition_proj = nn.Linear(d_model, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                  nn.SiLU(), nn.Linear(width, width))
            self.blocks.append(block)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, text_dim))

        self.register_buffer("alpha_bar", cosine_schedule(num_steps))

    def time_embedding(self, steps):
        """
        Sinusoidal embedding of the noise step, as in Transformer positions.

        Args:
            steps: (N,) noise step of each vector

        Returns:
            (N, width)
        """
        half = self.width // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=steps.device) / half)
        angles = steps.float().unsqueeze(-1) * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def predict_noise(self, noisy, steps, condition):
        """
        Predict the noise inside a noisy vector.

        Args:
            noisy: (N, 768) noisy vectors
            steps: (N,) noise step of each vector
            condition: (N, d_model) encoded token of each region

        Returns:
            predicted noise: (N, 768)
        """
        hidden = (self.input_proj(noisy) + self.condition_proj(condition)
                  + self.time_mlp(self.time_embedding(steps)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)

    def loss(self, condition, target, generator=None):
        """
        Noise-prediction loss (mean squared error) for a set of regions.

        Args:
            condition: (N, d_model) encoded tokens of hidden regions
            target: (N, 768) true text vectors of those regions
            generator: random generator (fixed for validation, so epochs compare)

        Returns:
            loss: scalar tensor
        """
        clean = F.normalize(target, dim=-1) * self.radius
        steps = torch.randint(0, self.num_steps, (len(clean),), device=clean.device,
                              generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)

        signal = self.alpha_bar[steps].unsqueeze(-1)
        noisy = signal.sqrt() * clean + (1 - signal).sqrt() * noise
        return F.mse_loss(self.predict_noise(noisy, steps, condition), noise)

    @torch.no_grad()
    def sample(self, condition, num_samples=10, sample_steps=50, generator=None):
        """
        Draw several possible text vectors for each region (DDIM, no extra noise).

          - each step: guess the clean vector, keep it on the sphere of radius sqrt(768)
            (true vectors all have that length), then move to the next, less noisy step

        Args:
            condition: (N, d_model) encoded tokens
            num_samples: samples per region
            sample_steps: denoising steps
            generator: random generator for the start noise

        Returns:
            samples: (N, num_samples, 768), each with length 1
        """
        num_regions = len(condition)
        condition = condition.repeat_interleave(num_samples, dim=0)
        vectors = torch.randn(len(condition), self.text_dim, device=condition.device,
                              generator=generator)

        schedule = torch.linspace(self.num_steps - 1, 0, sample_steps).long().tolist()
        for i, step in enumerate(schedule):
            signal = self.alpha_bar[step]
            steps = torch.full((len(vectors),), step, device=vectors.device, dtype=torch.long)
            noise = self.predict_noise(vectors, steps, condition)

            clean = (vectors - (1 - signal).sqrt() * noise) / signal.sqrt()
            clean = F.normalize(clean, dim=-1) * self.radius

            if i == len(schedule) - 1:
                vectors = clean
            else:
                next_signal = self.alpha_bar[schedule[i + 1]]
                vectors = next_signal.sqrt() * clean + (1 - next_signal).sqrt() * noise

        vectors = F.normalize(vectors, dim=-1)
        return vectors.view(num_regions, num_samples, self.text_dim)


if __name__ == "__main__":
    # quick check: PYTHONPATH=src python -m silence_ct.models.diffusion_head
    encoded = torch.randn(2, 9, 256)
    print("regression:", tuple(RegressionHead()(encoded).shape))     # expect (2, 9, 768)

    head = DiffusionHead()
    condition = torch.randn(5, 256)
    target = torch.randn(5, 768)
    print("diffusion loss:", round(head.loss(condition, target).item(), 3))   # about 1.0 at start
    samples = head.sample(condition, num_samples=4, sample_steps=10)
    print("samples:", tuple(samples.shape))                           # expect (5, 4, 768)
