# main.py

from src.data_loader import load_and_preprocess_data
from src.tree_builder import build_tree
from src.predictor import predict_batch
import pprint

from src.utils import accuracy

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/train.csv")

# Step 2: Build the decision tree
tree = build_tree(X_train, y_train.values, max_depth=3)

# Optional: print tree structure
print("Learned Decision Tree:")
pprint.pprint(tree)

# Step 3: Predict on test set
y_pred = predict_batch(tree, X_test)

# Step 4: Evaluate
acc = accuracy(y_test.values, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
