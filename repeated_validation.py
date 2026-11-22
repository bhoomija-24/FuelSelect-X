
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score

# 1. Load dataset
data = pd.read_csv("data/engineered_car_data.csv")

# 2. Keep Petrol and Diesel only
data = data[
    data["Fuel_Type"].isin(["Petrol", "Diesel"])
].copy()

# 3. Select features and target
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

# 4. Load selected model pipeline
model = joblib.load("models/best_fuel_model.pkl")

# 5. Repeated stratified cross-validation
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# 6. Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": make_scorer(
        f1_score,
        pos_label="Diesel"
    )
}

# 7. Evaluate model
results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

# 8. Display results
print("\nREPEATED 5-FOLD CROSS-VALIDATION")
print("Number of repetitions: 10")
print("Total evaluations: 50")

print("\nMODEL PERFORMANCE")

for metric in scoring:

    values = results["test_" + metric]

    print(f"\n{metric.upper()}")
    print(f"Mean: {values.mean():.4f}")
    print(f"Standard Deviation: {values.std():.4f}")
    print(f"Minimum: {values.min():.4f}")
    print(f"Maximum: {values.max():.4f}")

# 9. Save individual fold results
results_df = pd.DataFrame({
    "Accuracy": results["test_accuracy"],
    "Balanced_Accuracy": results["test_balanced_accuracy"],
    "F1_Score": results["test_f1"]
})

results_df.to_csv(
    "data/repeated_validation_results.csv",
    index=False
)

# 10. Plot balanced accuracy distribution
plt.figure(figsize=(9, 5))

plt.boxplot(
    results_df["Balanced_Accuracy"],
    tick_labels=["Logistic Regression"]
)

plt.ylabel("Balanced Accuracy")
plt.title("Repeated 5-Fold Cross-Validation")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "data/repeated_validation.png",
    dpi=300
)

plt.show()

print("\nREPEATED VALIDATION COMPLETED!")
