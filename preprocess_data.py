
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load engineered dataset
df = pd.read_csv("data/engineered_car_data.csv")

print("DATASET SHAPE:", df.shape)

# Separate target variable
y = df["Fuel_Type"]

# Select input features
# Avoid using both Year and Vehicle_Age
# Avoid price-derived features in this initial model
numerical_features = [
    "Vehicle_Age",
    "Kms_Driven",
    "Owner"
]

categorical_features = [
    "Car_Name",
    "Seller_Type",
    "Transmission"
]

selected_features = numerical_features + categorical_features

X = df[selected_features]

print("\nINPUT FEATURES:")
print(X.head())

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Fit and transform input data
X_processed = preprocessor.fit_transform(X)

# Save preprocessing pipeline
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

# Save processed features
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

np.save("data/X_processed.npy", X_processed)
np.save("data/y_target.npy", y.to_numpy())

print("\nPREPROCESSING COMPLETED!")
print("Original feature shape:", X.shape)
print("Processed feature shape:", X_processed.shape)
print("Target shape:", y.shape)

print("\nSaved files:")
print("models/preprocessor.pkl")
print("data/X_processed.npy")
print("data/y_target.npy")
