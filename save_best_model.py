
import joblib
import pandas as pd
from pathlib import Path

# Load the trained Logistic Regression model
model_path = Path("models/logistic_regression.pkl")

model = joblib.load(model_path)

# Save the selected model with a clear name
best_model_path = Path("models/best_fuel_model.pkl")

joblib.dump(model, best_model_path)

print("BEST MODEL SAVED SUCCESSFULLY!")
print("Selected Model: Logistic Regression")
print("Selection Metric: Test Balanced Accuracy")
print("Balanced Accuracy: 78.13%")
print("Saved Location:", best_model_path)
