"""
Losses

Training losses for predicting the text vectors of hidden regions. Only hidden
regions are scored: visible regions are inputs, and silent regions have no answer.

    cosine_loss:    1 - cosine(predicted, target), for the regression head
    diffusion_loss: noise prediction MSE, for the diffusion head (TODO)
"""

import torch.nn.functional as F


def cosine_loss(predicted, target, is_hidden):
    """
    Average (1 - cosine similarity) between predicted and true text vectors.

    Cosine compares the direction of two vectors (CT-CLIP vectors are compared
    this way), so 0 = same direction, 1 = unrelated, 2 = opposite:
      - hidden region   -> counted in the loss
      - visible region  -> ignored (the model already sees its text)
      - silent region   -> ignored (no true text)

    Args:
        predicted: (B, 9, 768) predicted text vectors
        target: (B, 9, 768) true text vectors (zeros where not hidden)
        is_hidden: (B, 9) True for hidden regions

    Returns:
        loss: scalar tensor
    """
    cosine = F.cosine_similarity(predicted, target, dim=-1)    # (B, 9)
    return (1 - cosine[is_hidden]).mean()


# TODO: diffusion_loss