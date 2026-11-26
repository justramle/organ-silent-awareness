"""
Evaluation on the test set (1,564 RadGenome validation scans, used once).

Five levels, each answering one question:

    1. Does it work?          cosine similarity to the hidden sentence, retrieval accuracy;
                              ABNORMAL / NORMAL: precision, recall, F1, confusion matrix, ROC-AUC
    2. Does it beat baselines? diffusion vs regression head; geometry bias vs content only
    3. Right reason?           flip test: blank one region's image vector and check whether
                              the predicted sentence changes for that region and its neighbors
    4. Trusted uncertainty?    does a larger spread of diffusion samples mean a larger error?
    5. Plan for next step?        results per region, including rare written regions
                              (thyroid, esophagus); later, NLST over time timeseries

Usage:
    PYTHONPATH=src python -m silence_ct.evaluate --config configs/config.yaml --checkpoint <path>
"""
