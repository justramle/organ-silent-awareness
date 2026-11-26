"""
Transformer encoder over region tokens.

A stack of standard pre-norm transformer blocks (attention + MLP). The only change from a
standard block is that attention scores receive the geometry bias from geometry_bias.py.

Input:  tokens (B, R, d), distance (B, R, R), touching (B, R, R)
Output: context-aware tokens (B, R, d)

After the encoder, the token of a hidden region carries information from its own image,
its neighbors, and the written parts of the report.
"""
