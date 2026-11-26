"""
Training script.

    1. read the config (paths, model size, head type, bias type, silence level)
    2. build train and small-validation datasets (train = 5,000 RadGenome train scans,
       small validation = 10% of train, used for early stopping only)
    3. build the model and optimizer (AdamW)
    4. each epoch: hide sentences, predict them, compute the loss, update
    5. save the best checkpoint by small-validation loss

The test set (1,564 RadGenome validation scans) is never used here.

Usage:
    PYTHONPATH=src python -m silence_ct.train --config configs/config.yaml
"""
