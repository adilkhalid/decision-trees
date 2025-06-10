# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Split features and label
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
