"""
report_generator.py - Generate Markdown reports of the pipeline runs.

This module automates the creation of markdown summaries detailing the dataset, 
models trained, comparison statistics, and explanations.
"""

import os
import pandas as pd
from typing import Dict, Any

def generate_markdown_report(
    df_shape: tuple,
    class_counts: Dict[str, int],
    results_df: pd.DataFrame,
    best_name: str,
    best_cm: Any,
    explanation_text: str,
    report_dir: str = "reports"
) -> tuple:
    """
    Generate and save a Markdown report inside the reports/ folder.

    Parameters
    ----------
    df_shape : tuple
        (rows, columns) of the loaded DataFrame.
    class_counts : dict
        {"malignant": int, "benign": int} counts.
    results_df : pandas.DataFrame
        Table of metrics for all trained models.
    best_name : str
        Name of the best performing model.
    best_cm : numpy.ndarray
        Confusion matrix of the best model.
    explanation_text : str
        Markdown explanation text from explanation_agent.py.
    report_dir : str
        Target folder to save the report.

    Returns
    -------
    report_path : str
        Absolute path to the saved markdown report.
    report_content : str
        The full string content of the markdown report.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Render results table as Markdown
    results_md = results_df.to_markdown(index=False)

    report_content = f"""# 🧬 CancerML Agent Pipeline Run Report

This report summarizes the performance evaluation of the machine learning pipeline on the Breast Cancer Wisconsin dataset.

---

## 📊 Dataset Information
* **Total Samples:** {df_shape[0]}
* **Number of Features:** {df_shape[1] - 1} (30 diagnostic features)
* **Target Classes:** 
  * Malignant (0): {class_counts.get('malignant', 0)} samples
  * Benign (1): {class_counts.get('benign', 0)} samples

---

## 🛠️ Models Trained
The pipeline trained and evaluated three classical models:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**

---

## 📈 Model Comparison Table
The models were evaluated on an 80/20 train/test stratified split.

{results_md}

*Note: Models are compared using macro-averaged metrics.*

---

## 🏆 Final Model Selection
* **Selected Best Model:** **{best_name}**
* **Selection Criteria:** Highest macro-averaged F1-Score.

---

## 🔢 Confusion Matrix of the Best Model
```
[[{best_cm[0][0]} {best_cm[0][1]}]
 [{best_cm[1][0]} {best_cm[1][1]}]]
```

---

{explanation_text}

---

## 📄 Disclaimer
This project is for educational and research purposes only and is not intended for medical diagnosis. Always consult a qualified healthcare professional.
"""

    report_path = os.path.join(report_dir, "pipeline_run_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Generator] Report successfully written to {report_path}")
    return report_path, report_content
