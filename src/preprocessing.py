"""
preprocessing.py - Prepare the data for model training.

This module handles:
  1. Splitting the dataset into training and testing sets.
  2. Scaling (standardising) the features so every feature has
     mean ~ 0 and standard deviation ~ 1.  This is especially
     important for algorithms like Logistic Regression and SVM
     that are sensitive to feature magnitudes.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'target' column and feature columns.
    test_size : float
        Fraction of data reserved for testing (default 20 %).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
    """
    # Separate features (X) from the label (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Perform a stratified train/test split to keep class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keeps the malignant/benign ratio equal in both sets
    )

    print(f"[OK] Data split: {len(X_train)} train / {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardise features using StandardScaler.

    The scaler is fit ONLY on the training set to avoid data leakage,
    then applied to both the training and test sets.

    Parameters
    ----------
    X_train, X_test : array-like
        Raw feature matrices.

    Returns
    -------
    X_train_scaled, X_test_scaled : numpy.ndarray
        Scaled feature matrices.
    scaler : StandardScaler
        The fitted scaler (useful if you want to transform new data later).
    """
    scaler = StandardScaler()

    # Fit on train data, then transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[OK] Features scaled (StandardScaler)")
    return X_train_scaled, X_test_scaled, scaler
