
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. Load engineered dataset
# --------------------------------

df = pd.read_csv("data/engineered_car_data.csv")

# Keep only Petrol and Diesel for initial experiment
df = df[df["Fuel_Type"].isin(["Petrol", "Diesel"])].copy()

print("DATASET SHAPE:", df.shape)

# --------------------------------
# 2. Select input and target
# --------------------------------

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

features = numerical_features + categorical_features

X = df[features]
y = df["Fuel_Type"]

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())

# --------------------------------
# 3. Split the dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# --------------------------------
# 4. Create preprocessing pipeline
# --------------------------------

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

# --------------------------------
# 5. Define machine learning models
# --------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

results = []

# --------------------------------
# 6. Train and evaluate models
# --------------------------------

for name, model in models.items():

    print("\n" + "=" * 50)
    print("MODEL:", name)
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict test data
    predictions = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)

    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    precision = precision_score(
        y_test, predictions,
        pos_label="Diesel",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        pos_label="Diesel",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        pos_label="Diesel",
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Diesel Precision:", round(precision, 4))
    print("Diesel Recall:", round(recall, 4))
    print("Diesel F1-score:", round(f1, 4))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("CONFUSION MATRIX:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Petrol", "Diesel"]
    ))

    # Save trained pipeline
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        f"models/{filename}.pkl"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Diesel_Precision": precision,
        "Diesel_Recall": recall,
        "Diesel_F1": f1
    })

# --------------------------------
# 7. Compare model performance
# --------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced_Accuracy",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

print("\nMODEL TRAINING COMPLETED!")
print("Trained models saved in models folder.")
print("Comparison saved in data/model_comparison.csv")
