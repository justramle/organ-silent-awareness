"""
Losses

Training losses for predicting the text vectors of hidden regions. Only hidden
regions are scored: visible regions are inputs, and silent regions have no answer.

    cosine_loss:    1 - cosine(predicted, target), for the regression head
    state_loss:     binary cross-entropy, for the state head
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


def state_loss(abnormal_logit, state, is_hidden):
    """
    Binary cross-entropy between the predicted and true normal/abnormal state.

    Only hidden regions are scored, as in cosine_loss:
      - hidden ABNORMAL region -> label 1
      - hidden NORMAL region   -> label 0
      - visible / silent       -> ignored

    Args:
        abnormal_logit: (B, 9) logit of "abnormal" from the state head
        state: (B, 9) 1 ABNORMAL, 0 NORMAL, -1 SILENT
        is_hidden: (B, 9) True for hidden regions

    Returns:
        loss: scalar tensor
    """
    labels = (state[is_hidden] == 1).float()
    return F.binary_cross_entropy_with_logits(abnormal_logit[is_hidden], labels)


# TODO: diffusion_loss