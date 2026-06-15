"""
evaluation.py - Evaluate trained models and pick the best one.

For every model this module computes:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion matrix

Results are printed to the console and saved as a CSV report.
The best model is selected by highest **F1-score** (macro-averaged).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_models(models, X_test, y_test, report_dir="reports"):
    """
    Evaluate every model on the test set and select the best one.

    Parameters
    ----------
    models : dict
        {model_name: fitted_estimator} as returned by train_models().
    X_test : array-like
        Scaled test features.
    y_test : array-like
        True test labels.
    report_dir : str
        Folder where the CSV summary will be saved.

    Returns
    -------
    best_name : str
        Name of the model with the highest F1-score.
    best_model : estimator
        The best fitted model object.
    results_df : pandas.DataFrame
        Table of metrics for every model.
    """
    # Create the reports directory if it doesn't exist yet
    os.makedirs(report_dir, exist_ok=True)

    results = []  # will hold one dict per model

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)

    for name, model in models.items():
        # Generate predictions
        y_pred = model.predict(X_test)

        # Compute metrics  (macro average treats both classes equally)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        # Store metrics for the summary table
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
        })

        # Pretty-print the results for this model
        print(f"\n[MODEL] {name}")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   Confusion Matrix:\n{_indent_matrix(cm)}")
        print(f"\n   Classification Report:\n"
              f"{classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])}")

    # -- Build a summary DataFrame and pick the best model --
    results_df = pd.DataFrame(results)

    best_idx = int(np.argmax(results_df["F1-Score"].values))
    best_name = results_df.loc[best_idx, "Model"]
    best_model = models[best_name]

    # Print the comparison table to the console
    print("\n" + "-" * 60)
    print("  MODEL COMPARISON TABLE")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)

    print("\n" + "=" * 60)
    print(f">>> Best Model: {best_name}  (F1-Score = {results_df.loc[best_idx, 'F1-Score']})")
    print("=" * 60)

    # Save the summary to CSV inside the reports/ folder
    report_path = os.path.join(report_dir, "model_comparison.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\n[SAVED] Report -> {report_path}")

    return best_name, best_model, results_df


# -- helper --
def _indent_matrix(matrix, indent=6):
    """Return a string representation of a numpy matrix with left indent."""
    lines = str(matrix).split("\n")
    return "\n".join(" " * indent + line for line in lines)
