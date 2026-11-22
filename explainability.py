
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Load dataset
data = pd.read_csv("data/engineered_car_data.csv")

# Keep only Petrol and Diesel
data = data[data["Fuel_Type"].isin(["Petrol", "Diesel"])].copy()

# Select input features
features = [
    "Vehicle_Age",
    "Kms_Driven",
    "Owner",
    "Car_Name",
    "Seller_Type",
    "Transmission"
]

X = data[features]
y = data["Fuel_Type"]

# Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the selected model
model = joblib.load("models/best_fuel_model.pkl")

# Calculate permutation importance
results = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="balanced_accuracy",
    n_repeats=20,
    random_state=42
)

# Create results table
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": results.importances_mean,
    "Standard_Deviation": results.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE RESULTS")
print(importance_df.to_string(index=False))

# Save results
importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    xerr=importance_df["Standard_Deviation"]
)

plt.xlabel("Decrease in Balanced Accuracy")
plt.ylabel("Vehicle Features")
plt.title("FuelSelect-X - Permutation Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "data/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed!")
