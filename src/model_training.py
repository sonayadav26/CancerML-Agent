"""
model_training.py - Train multiple classification models.

This module trains three classical ML models on the breast cancer dataset:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Random Forest Classifier

Each model is returned inside a dictionary so the evaluation module
can iterate over them conveniently.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def train_models(X_train, y_train):
    """
    Train three classifiers and return them in a dictionary.

    Parameters
    ----------
    X_train : array-like
        Scaled training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    models : dict
        Keys are human-readable model names; values are fitted estimators.
    """
    # Define the models with sensible defaults
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=10_000,       # ensure convergence on this dataset
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",          # radial-basis-function kernel (default)
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,      # 100 decision trees
            random_state=42,
        ),
    }

    # Train each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[OK] {name} - trained successfully")

    return models
