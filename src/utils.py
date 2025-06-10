import numpy as np


def accuracy(y_true, y_pred):
    """
    Compute accuracy: % of correct predictions.
    """
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct / len(y_true)
