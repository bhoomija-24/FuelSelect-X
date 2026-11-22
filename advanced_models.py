
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)

# ----------------------------------
# 1. Load dataset
# ----------------------------------

df = pd.read_csv("data/engineered_car_data.csv")

# Initial experiment: Petrol vs Diesel
df = df[df["Fuel_Type"].isin(["Petrol", "Diesel"])].copy()

# Encode target
y = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1
})

# Select features
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

# ----------------------------------
# 2. Train-test split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 3. Preprocessing
# ----------------------------------

preprocessor = ColumnTransformer([
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
])

# ----------------------------------
# 4. Define models
# ----------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# ----------------------------------
# 5. Stratified 5-fold cross-validation
# ----------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1"
}

results = []

print("\n5-FOLD CROSS VALIDATION RESULTS")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    mean_accuracy = np.mean(
        scores["test_accuracy"]
    )

    mean_balanced = np.mean(
        scores["test_balanced_accuracy"]
    )

    std_balanced = np.std(
        scores["test_balanced_accuracy"]
    )

    mean_f1 = np.mean(scores["test_f1"])

    print("\nMODEL:", name)
    print("Mean Accuracy:", round(mean_accuracy, 4))
    print("Mean Balanced Accuracy:", round(mean_balanced, 4))
    print("Balanced Accuracy Std:", round(std_balanced, 4))
    print("Mean F1-score:", round(mean_f1, 4))

    results.append({
        "Model": name,
        "Mean_Accuracy": mean_accuracy,
        "Mean_Balanced_Accuracy": mean_balanced,
        "Balanced_Accuracy_Std": std_balanced,
        "Mean_F1": mean_f1
    })

# ----------------------------------
# 6. Compare final test performance
# ----------------------------------

print("\nFINAL TEST SET RESULTS")

test_results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    balanced = balanced_accuracy_score(
        y_test, predictions
    )

    print("\nMODEL:", name)
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Balanced Accuracy:", round(balanced, 4))

    print(classification_report(
        y_test,
        predictions,
        target_names=["Petrol", "Diesel"],
        zero_division=0
    ))

    test_results.append({
        "Model": name,
        "Test_Accuracy": accuracy,
        "Test_Balanced_Accuracy": balanced
    })

# ----------------------------------
# 7. Save comparison
# ----------------------------------

cv_results = pd.DataFrame(results)
test_df = pd.DataFrame(test_results)

final_results = cv_results.merge(
    test_df,
    on="Model"
)

final_results.to_csv(
    "data/advanced_model_comparison.csv",
    index=False
)

print("\nFINAL COMPARISON")
print(final_results.to_string(index=False))

print("\nADVANCED MODEL EVALUATION COMPLETED!")
