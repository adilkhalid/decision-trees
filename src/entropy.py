# src/entropy.py

import numpy as np


def entropy(y):
    """
    Calculate entropy of a label array.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(y, y_left, y_right):
    """
    Compute the information gain from a binary split.
    """
    H_parent = entropy(y)

    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    H_children = weight_left * entropy(y_left) + weight_right * entropy(y_right)

    return H_parent - H_children
