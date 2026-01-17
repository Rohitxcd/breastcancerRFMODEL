import pickle
import pandas as pd

# Load model
with open("random_forest_breast_cancer.pkl", "rb") as f:
    rfc = pickle.load(f)

# Load data again
df = pd.read_csv("data/breastcancer.csv")
df = df.drop("id", axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

X = df.drop("diagnosis", axis=1)

# Predict
predictions = rfc.predict(X)
print(predictions)

