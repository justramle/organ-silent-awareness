"""
Retrieval: from a predicted sentence vector to a real sentence.

The heads predict a vector, not words. To read the result, we search the bank of real
training sentences (CT-CLIP vectors) for the closest one, using cosine similarity.

Retrieval has no trainable parameters. It is also used in evaluation: if the retrieved
sentence matches the hidden sentence (or has the same ABNORMAL / NORMAL meaning), the
prediction is counted as correct.
"""
