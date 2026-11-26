# Geometry from organ masks: a toy example

A **voxel equal to 1** is a voxel that SAT marked as "this is the heart". All other voxels are 0, meaning "not the heart".

Here is a very small example: one slice of 6 × 6 voxels, each voxel 1 mm (a real mask is 3D and much bigger).

## Heart mask and lung mask

```
      HEART MASK                     LUNG MASK
   x→  0 1 2 3 4 5               x→  0 1 2 3 4 5
y=0    0 0 0 0 0 0           y=0     0 0 0 0 0 0
y=1    0 0 0 0 0 0           y=1     1 1 0 0 0 0
y=2    0 0 1 1 0 0           y=2     1 1 0 0 0 0
y=3    0 0 1 1 0 0           y=3     1 1 0 0 0 0
y=4    0 0 0 0 0 0           y=4     0 0 0 0 0 0
y=5    0 0 0 0 0 0           y=5     0 0 0 0 0 0
```

- The heart mask has **4 voxels equal to 1**, at positions (2,2), (3,2), (2,3), (3,3).
- The lung mask has **6 voxels equal to 1**, in columns x = 0 and x = 1, rows y = 1 to 3.

## 1. Center: where is the organ?

Take the **mean position** of the voxels equal to 1.

```
Heart:  x = (2 + 3 + 2 + 3) / 4 = 2.5
        y = (2 + 2 + 3 + 3) / 4 = 2.5      → heart center = (2.5, 2.5) mm

Lung:   x = (0+1+0+1+0+1) / 6 = 0.5
        y = (1+1+2+2+3+3) / 6 = 2.0        → lung center  = (0.5, 2.0) mm
```

## 2. Volume: how big is the organ?

Count the voxels equal to 1, then multiply by the size of one voxel.

```
Heart:  4 voxels × 1 mm² = 4 mm²
Lung:   6 voxels × 1 mm² = 6 mm²
```

In real 3D data, each voxel is a small block (for example 5 mm³), so the result is a volume, given in mL.

## 3. Touching: are the two organs neighbors?

**Grow** the heart mask by 1 voxel in every direction, then check whether it overlaps the lung.

```
   HEART AFTER GROWING BY 1 VOXEL     (compared with the LUNG)
   x→  0 1 2 3 4 5
y=1    0 0 1 1 0 0
y=2    0 1 1 1 1 0     ← voxel (1,2) is now 1, and the lung also has voxel (1,2) = 1
y=3    0 1 1 1 1 0     ← voxel (1,3) also overlaps the lung
y=4    0 0 1 1 0 0
```

The grown heart **overlaps** lung voxels, so **touching = 1** (they touch).

If the lung were far away, for example only in column x = 0 with the heart in columns 3–4, growing by 1 voxel would not reach it, and **touching = 0**.

## 4. Distance: how far apart are they?

The distance between the **two centers**, using the Pythagorean theorem.

```
heart center = (2.5, 2.5)
lung center  = (0.5, 2.0)

distance = √( (2.5 − 0.5)² + (2.5 − 2.0)² )
         = √( 4 + 0.25 )
         ≈ 2.06 mm
```

## Summary for this example

| | Heart | Lung |
| --- | --- | --- |
| voxels equal to 1 | 4 | 6 |
| center | (2.5, 2.5) | (0.5, 2.0) |
| volume | 4 mm² | 6 mm² |
| touching heart–lung | **1** (touch) | |
| distance heart–lung | **2.06 mm** | |

The geometry step in the pipeline does exactly these 4 computations, only on the real 3D volume (for example 368 × 368 × 63 voxels) and for all 9 organs of each scan.

---

# How the model uses these numbers: the geometry bias

For two organs *i* and *j*, a bias is added to the attention score:

```
bias(i, j) =   − w · dist(i, j)     +     b · touch(i, j)
               └── signal 1 ──┘           └── signal 2 ──┘
                  distance                 touching or not
```

## What does the "−" sign mean?

The minus sign is in front of the distance, because **the farther away, the lower the score**:

- far (large dist) → a lot is subtracted → **low** bias → the model attends **less**
- near (small dist) → little is subtracted → **higher** bias → the model attends **more**

The touching part has a **+** sign, because **touching organs get extra score**.

## Example with numbers

Take `w = 0.01` and `b = 2` (example values only), and look from the heart to 3 organs:

| Organ | dist | touch | − w · dist | + b · touch | bias |
| --- | --- | --- | --- | --- | --- |
| esophagus | 28 mm | 1 | −0.28 | +2 | **+1.72** |
| lung | 100 mm | 1 | −1.00 | +2 | **+1.00** |
| thyroid | 185 mm | 0 | −1.85 | 0 | **−1.85** |

The higher the bias, the more the heart "listens" to that organ.

## Why do we need both? Look at the lung row

The lung is **very large**, so its center is far from the heart's center (100 mm), even though the two organs **touch**.

```
        lung (large)             heart
   ┌──────────────────┐      ┌──────┐
   │                  │      │      │
   │        ●─────────┼──────┼──●   │     ● = center
   │   lung center    │      │center│
   └──────────────────┘◀touch▶└─────┘
        centers are 100 mm apart, but the organs are side by side
```

- With **distance only**: the lung gets −1.00, and the model thinks the lung is far from the heart. That is wrong.
- Adding **touch**: the lung gets +2, so the bias becomes +1.00, and the model knows the lung is a neighbor. That is right.

So touching corrects the cases where the distance between two centers is misleading.

In the other direction, with **touch only**: the trachea and the thyroid are both 0, so the model cannot tell which one is closer. Distance separates those two.

## Summary

| Signal | Sign | What it helps with | Weakness |
| --- | --- | --- | --- |
| dist | − (farther = lower) | separates near from far for every pair | misleading for large organs such as the lung |
| touch | + (touching = higher) | identifies direct neighbors for certain | only 0 or 1, does not say how far |

The two signals **complement each other**, so the formula uses both. The values of `w` and `b` are **learned by the model** during training.

Whether the geometry bias improves the results is a hypothesis, not yet shown. It is tested by an ablation: content only, + touch, + dist, + both.
