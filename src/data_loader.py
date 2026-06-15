"""
data_loader.py - Load the Breast Cancer Wisconsin dataset.

This module loads the built-in breast cancer dataset from scikit-learn
and returns it as a pandas DataFrame for easy inspection and manipulation.
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing 30 feature columns and a 'target' column.
        target = 0 -> malignant, target = 1 -> benign.
    """
    # Load the dataset bundled with scikit-learn
    cancer = load_breast_cancer()

    # Build a DataFrame from the feature matrix
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

    # Append the target column (0 = malignant, 1 = benign)
    df["target"] = cancer.target

    # Print the dataset shape explicitly
    print(f"[OK] Dataset loaded successfully!")
    print(f"   Shape : {df.shape}  ({df.shape[0]} samples, {df.shape[1]} columns)")
    print(f"   Features : {df.shape[1] - 1}")
    print(f"   Classes  : 0 (malignant) = {(df['target'] == 0).sum()}, "
          f"1 (benign) = {(df['target'] == 1).sum()}")
    print(f"\n   First 5 rows (preview):")
    print(df.head().to_string(max_cols=6))

    return df
