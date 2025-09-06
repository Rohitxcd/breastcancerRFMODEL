

# Breast Cancer Prediction using XAI

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project focuses on building a **Random Forest model** to predict **breast cancer** (malignant or benign) and explain its predictions using **SHAP (Explainable AI)**.
The goal is not only high accuracy but also **interpretability**, helping doctors understand the model's decisions.

---

## Dataset

* **File:** `data/breastcancer.csv`
* **Instances:** 569
* **Features:** 30 numeric features (e.g., radius, texture, smoothness)
* **Target:** `diagnosis` (M = malignant, B = benign)
* **Source:** (Add your dataset link here if available)

---

## Project Tasks

1. **Data Preprocessing:**

   * Handle missing values
   * Encode categorical variables
   * Feature scaling

2. **Modeling:**

   * Train **Random Forest classifier**
   * Evaluate using metrics: Accuracy, Precision, Recall, F1-Score

3. **Explainability (XAI):**

   * Apply **SHAP** to explain model predictions
   * Visualize feature importance and decision impact

---

## Folder Structure

```
├── data/
│   └── breastcancer.csv
├── plots/
│   └── shap_plot.png
├── sampledataset.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

---

## Usage

Run the main script to train the model and generate explanations:

```bash
python sampledataset.py
```

**Example Output:**

* Accuracy: **96.49%**
* SHAP summary plot showing top features influencing predictions

**SHAP Plot Example:**
![SHAP Plot](plots/shap_plot.png)

---

## Results

* Model achieved **96.49% accuracy** on the test set.
* Top features impacting prediction: `radius_mean`, `concave points_mean`, `texture_mean`.
* SHAP plot provides **visual explanation** of predictions for individual patients.

---

## Future Work

* Experiment with more advanced models (LightGBM, CatBoost)
* Deploy as a **web app** for interactive explanations
* Extend to other medical datasets for generalizability

---

## References

1. UCI Breast Cancer Dataset: [link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
2. SHAP Documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)

---

## License

This project is licensed under the **MIT License**.


