# src/predictor.py

def predict(tree, sample):
    """
    Predict a label for a single sample using the decision tree.
    """
    if "leaf" in tree:
        return tree["prediction"]

    feature = tree["feature"]
    threshold = tree["threshold"]

    if sample[feature] <= threshold:
        return predict(tree["left"], sample)
    else:
        return predict(tree["right"], sample)


def predict_batch(tree, X):
    """
    Predict labels for a batch of samples (pandas DataFrame).
    """
    return [predict(tree, row) for _, row in X.iterrows()]
