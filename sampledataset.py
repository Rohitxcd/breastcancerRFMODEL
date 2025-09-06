# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 2. Load your dataset
df = pd.read_csv("data[2].csv")

# 3. Preprocess
df = df.drop("id", axis=1)  # drop ID column

# Encode target: M -> 1, B -> 0
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

# 4. Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Random Forest
rfc = RandomForestClassifier(n_estimators=300, random_state=42)
rfc.fit(X_train, y_train)

# 7. Predictions
predictions = rfc.predict(X_test)

# 8. Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
acc = accuracy_score(y_test, predictions)
print("\nAccuracy of Random Forest Model: {:.2f}%".format(acc*100))

# 9. Feature importance
feat_imp = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features - Random Forest")
plt.show()

# 10. SHAP Explainability
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values[1], X_test)

# Explain a single prediction
idx = 0  # first test sample
shap.force_plot(explainer.expected_value[1], shap_values[1][idx], X_test.iloc[idx], matplotlib=True)

