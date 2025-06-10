# src/tree_builder.py

import numpy as np
from .entropy import information_gain

def best_split_for_feature(X_col, y):
    best_gain = -1
    best_threshold = None
    thresholds = np.unique(X_col)

    for threshold in thresholds:
        left_mask = X_col <= threshold
        right_mask = X_col > threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        gain = information_gain(y, y_left, y_right)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = -1

    for feature in X.columns:
        X_col = X[feature].values
        threshold, gain = best_split_for_feature(X_col, y)

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_gain


def build_tree(X, y, depth=0, max_depth=3):
    # Base cases
    if len(set(y)) == 1:
        return {"leaf": True, "prediction": y[0]}

    if depth >= max_depth or len(y) < 2:
        majority_class = np.bincount(y).argmax()
        return {"leaf": True, "prediction": majority_class}

    # Find best split
    feature, threshold, gain = best_split(X, y)
    if gain == 0:
        majority_class = np.bincount(y).argmax()
        return {"leaf": True, "prediction": majority_class}

    # Split the data
    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    # Recursively build tree
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X_left, y_left, depth + 1, max_depth),
        "right": build_tree(X_right, y_right, depth + 1, max_depth)
    }
