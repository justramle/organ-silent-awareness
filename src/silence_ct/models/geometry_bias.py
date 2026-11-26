"""
Geometry bias for attention.

Standard attention lets every region look at every other region based only on content.
We add a bias computed from anatomy, so a region attends more to its real neighbors:

    bias[h, i, j] = -w_h * distance[i, j] + b_h * touching[i, j]

    distance   center-to-center distance between regions i and j (mm, normalized)
    touching   1 if regions i and j share a boundary
    w_h, b_h   learned per attention head (w_h >= 0)

The minus sign means far regions get a lower score. Each head learns its own w_h and b_h,
so some heads can stay local while others look at all regions.

The bias is added to the attention scores before softmax:

    attention = softmax(Q K^T / sqrt(d) + bias)

Ablation (hypothesis H2): content only / + touching / + distance / both.
"""
