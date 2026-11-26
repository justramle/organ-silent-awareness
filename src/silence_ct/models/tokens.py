"""
Input tokens.

Every region becomes one token. A token combines:

    - the region's image vector (OrganLens, 1024-d)
    - the region's text vector (CT-CLIP, 768-d), or a learned [HIDDEN] vector when the
      region's text is hidden or silent
    - a learned region embedding, so the model knows which organ the token is

Image and text vectors are projected to the same model width d and added together.
Output: (B, R, d), one token per region.
"""
