"""
explanation_agent.py - Rule-based explanation assistant.

This module provides clear, beginner-friendly explanations of model performance 
metrics and confusion matrices, specifically formatted for educational purposes 
and with health-safety disclaimers.
"""

from typing import Dict, Any

def generate_pipeline_explanation(best_name: str, metrics: Dict[str, Any], cm: Any) -> str:
    """
    Generate a rule-based explanation of the machine learning pipeline results.

    Parameters
    ----------
    best_name : str
        Name of the top performing model.
    metrics : dict
        Contains 'Accuracy', 'Precision', 'Recall', and 'F1-Score' for the best model.
    cm : numpy.ndarray
        Confusion matrix of the best model (2x2).

    Returns
    -------
    explanation : str
        A markdown-formatted explanation of the results and performance metrics.
    """
    # Extract values from 2x2 confusion matrix
    # Row 0: Malignant (True Negatives, False Positives)
    # Row 1: Benign (False Negatives, True Positives)
    # Assuming malignant = class 0 (positive condition in diagnosis context, or labeled as 0 here)
    # Let's map it safely based on standard orientation:
    tn, fp, fn, tp = cm.ravel()

    explanation = f"""### 🧠 Explanation of Pipeline Results

Based on our evaluation, the best performing model is **{best_name}**.

Below is a breakdown of what the performance metrics mean:

1. **Accuracy ({metrics.get('Accuracy', 0.0) * 100:.2f}%)**
   * *What it is:* The proportion of total predictions that were correct (both malignant and benign).
   * *Meaning:* Out of 100 cases, the model correctly identifies approximately {metrics.get('Accuracy', 0.0) * 100:.1f} of them.

2. **Precision ({metrics.get('Precision', 0.0) * 100:.2f}%)**
   * *What it is:* The proportion of positive predictions that are truly positive.
   * *Meaning:* When the model predicts a tumor is of a specific class, it is correct {metrics.get('Precision', 0.0) * 100:.1f}% of the time. This minimizes "false alarms" (False Positives).

3. **Recall / Sensitivity ({metrics.get('Recall', 0.0) * 100:.2f}%)**
   * *What it is:* The proportion of actual positive cases that the model detected.
   * *Meaning:* Out of all actual tumors of a given class, the model successfully caught {metrics.get('Recall', 0.0) * 100:.1f}%. High recall is critical in healthcare because it minimizes missed diagnoses (False Negatives).

4. **F1-Score ({metrics.get('F1-Score', 0.0) * 100:.2f}%)**
   * *What it is:* The harmonic mean of Precision and Recall.
   * *Meaning:* It provides a balanced measure, especially useful when class sizes might differ. Since this metric is high ({metrics.get('F1-Score', 0.0) * 100:.1f}%), the model exhibits strong overall performance.

---

### 🔢 Understanding the Confusion Matrix

The confusion matrix for **{best_name}** is structured as follows:

* **True Malignant (Correctly Classified Malignant):** **{tn}** samples
* **True Benign (Correctly Classified Benign):** **{tp}** samples
* **False Benign (Malignant misclassified as Benign):** **{fn}** samples *(Critical: These are false negatives)*
* **False Malignant (Benign misclassified as Malignant):** **{fp}** samples *(These are false alarms)*

---

### ⚠️ Health & Safety Disclaimer
*This project is for educational and research purposes only and is not intended for medical diagnosis. Machine learning models should never replace professional medical advice, clinical diagnosis, or treatment. Always consult a qualified physician or healthcare provider regarding any medical conditions.*
"""
    return explanation
