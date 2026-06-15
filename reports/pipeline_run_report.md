# 🧬 CancerML Agent Pipeline Run Report

This report summarizes the performance evaluation of the machine learning pipeline on the Breast Cancer Wisconsin dataset.

---

## 📊 Dataset Information
* **Total Samples:** 569
* **Number of Features:** 30 (30 diagnostic features)
* **Target Classes:** 
  * Malignant (0): 212 samples
  * Benign (1): 357 samples

---

## 🛠️ Models Trained
The pipeline trained and evaluated three classical models:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**

---

## 📈 Model Comparison Table
The models were evaluated on an 80/20 train/test stratified split.

| Model                  |   Accuracy |   Precision |   Recall |   F1-Score |
|:-----------------------|-----------:|------------:|---------:|-----------:|
| Logistic Regression    |     0.9825 |      0.9812 |   0.9812 |     0.9812 |
| Support Vector Machine |     0.9825 |      0.9812 |   0.9812 |     0.9812 |
| Random Forest          |     0.9561 |      0.9551 |   0.9504 |     0.9526 |

*Note: Models are compared using macro-averaged metrics.*

---

## 🏆 Final Model Selection
* **Selected Best Model:** **Logistic Regression**
* **Selection Criteria:** Highest macro-averaged F1-Score.

---

## 🔢 Confusion Matrix of the Best Model
```
[[41 1]
 [1 71]]
```

---

### 🧠 Explanation of Pipeline Results

Based on our evaluation, the best performing model is **Logistic Regression**.

Below is a breakdown of what the performance metrics mean:

1. **Accuracy (98.25%)**
   * *What it is:* The proportion of total predictions that were correct (both malignant and benign).
   * *Meaning:* Out of 100 cases, the model correctly identifies approximately 98.2 of them.

2. **Precision (98.12%)**
   * *What it is:* The proportion of positive predictions that are truly positive.
   * *Meaning:* When the model predicts a tumor is of a specific class, it is correct 98.1% of the time. This minimizes "false alarms" (False Positives).

3. **Recall / Sensitivity (98.12%)**
   * *What it is:* The proportion of actual positive cases that the model detected.
   * *Meaning:* Out of all actual tumors of a given class, the model successfully caught 98.1%. High recall is critical in healthcare because it minimizes missed diagnoses (False Negatives).

4. **F1-Score (98.12%)**
   * *What it is:* The harmonic mean of Precision and Recall.
   * *Meaning:* It provides a balanced measure, especially useful when class sizes might differ. Since this metric is high (98.1%), the model exhibits strong overall performance.

---

### 🔢 Understanding the Confusion Matrix

The confusion matrix for **Logistic Regression** is structured as follows:

* **True Malignant (Correctly Classified Malignant):** **41** samples
* **True Benign (Correctly Classified Benign):** **71** samples
* **False Benign (Malignant misclassified as Benign):** **1** samples *(Critical: These are false negatives)*
* **False Malignant (Benign misclassified as Malignant):** **1** samples *(These are false alarms)*

---

### ⚠️ Health & Safety Disclaimer
*This project is for educational and research purposes only and is not intended for medical diagnosis. Machine learning models should never replace professional medical advice, clinical diagnosis, or treatment. Always consult a qualified physician or healthcare provider regarding any medical conditions.*


---

## 📄 Disclaimer
This project is for educational and research purposes only and is not intended for medical diagnosis. Always consult a qualified healthcare professional.
