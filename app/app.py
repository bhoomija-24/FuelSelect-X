
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------
# PATHS AND MODEL LOADING
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_fuel_model.pkl"
DATA_PATH = BASE_DIR / "data" / "engineered_car_data.csv"

model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATA_PATH)

# Get exact categories from the training dataset
car_names = sorted(dataset["Car_Name"].dropna().unique().tolist())
seller_types = sorted(dataset["Seller_Type"].dropna().unique().tolist())
transmissions = sorted(dataset["Transmission"].dropna().unique().tolist())

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="FuelSelect-X | Smart Fuel Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

.hero {
    background: linear-gradient(120deg, #101c3a, #234b80);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 17px;
    margin-bottom: 0;
}

.section-title {
    color: #172b4d;
    font-size: 23px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1e8f0;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(120deg, #e5f8ef, #f3fff8);
    border: 1px solid #a7e5c2;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.result-card h2 {
    color: #087443;
    font-size: 30px;
}

.result-card p {
    color: #355c49;
    font-size: 16px;
}

div.stButton > button {
    background: linear-gradient(120deg, #2457a7, #3979d8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: 600;
    width: 100%;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #173f7c;
    color: white;
    border: none;
}

[data-testid="stSidebar"] {
    background: #eaf0f8;
}

[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.markdown("## 🚗 FuelSelect-X")

    st.write("Smart Vehicle Fuel Prediction")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "FuelSelect-X uses machine learning "
        "to classify a vehicle as Petrol or Diesel "
        "based on its details."
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Selected Model:** Logistic Regression")

    st.write("**Test Balanced Accuracy:** 78.13%")

    st.write("**Supported Classes:** Petrol and Diesel")

    st.warning(
        "CNG prediction is not supported because "
        "the original dataset contains only two CNG records."
    )

    st.divider()

    st.caption("Academic Machine Learning Project | 2026")

# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown("""
<div class="hero">
    <h1>🚗 FuelSelect-X</h1>
    <p>Intelligent Vehicle Fuel Type Prediction System</p>
    <p>Discover the predicted fuel category using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# OVERVIEW METRICS
# -----------------------------------------

st.markdown("### 📊 Project Overview")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Training Records", len(dataset))

with m2:
    st.metric("Fuel Categories", "2")

with m3:
    st.metric("Selected Algorithm", "Logistic Regression")

st.divider()

# -----------------------------------------
# VEHICLE INPUT FORM
# -----------------------------------------

st.markdown(
    '<div class="section-title">🚘 Enter Vehicle Details</div>',
    unsafe_allow_html=True
)

st.write("Provide the following information to generate a prediction.")

with st.form("fuel_prediction_form"):

    st.markdown("#### 1. Vehicle Information")

    col1, col2 = st.columns(2)

    with col1:
        car_name = st.selectbox(
            "Car Name",
            car_names,
            help="Select a vehicle name from the training dataset."
        )

        vehicle_age = st.number_input(
            "Vehicle Age (Years)",
            min_value=0,
            max_value=40,
            value=5,
            step=1
        )

    with col2:
        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=1000
        )

        owner = st.selectbox(
            "Previous Owners",
            [0, 1, 2, 3],
            format_func=lambda x: (
                "First Owner" if x == 0
                else f"{x} Previous Owner(s)"
            )
        )

    st.divider()

    st.markdown("#### 2. Additional Details")

    col3, col4 = st.columns(2)

    with col3:
        seller_type = st.selectbox(
            "Seller Type",
            seller_types
        )

    with col4:
        transmission = st.selectbox(
            "Transmission",
            transmissions
        )

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Predict Fuel Type",
        use_container_width=True
    )

# -----------------------------------------
# PREDICTION
# -----------------------------------------


# -----------------------------------------
# PREDICTION
# -----------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "Vehicle_Age": vehicle_age,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Car_Name": car_name,
        "Seller_Type": seller_type,
        "Transmission": transmission
    }])

    try:

        # Get prediction directly from the model
        prediction = model.predict(input_data)[0]

        # Support both text and numeric model outputs
        if str(prediction).lower() == "1":
            fuel_type = "Diesel"
        elif str(prediction).lower() == "0":
            fuel_type = "Petrol"
        else:
            fuel_type = str(prediction)

        st.divider()

        st.markdown("### 🎯 Prediction Result")

        st.markdown(f"""
        <div class="result-card">
            <h2>⛽ {fuel_type}</h2>
            <p>Predicted Fuel Category</p>
        </div>
        """, unsafe_allow_html=True)

        # Display probability scores
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            st.markdown("### 📈 Model Probability Scores")

            for class_value, probability in zip(classes, probabilities):

                label = str(class_value)

                if label == "0":
                    label = "Petrol"
                elif label == "1":
                    label = "Diesel"

                st.write(
                    f"**{label}: {probability * 100:.2f}%**"
                )

                st.progress(float(probability))

        st.info(
            "These probability scores are model estimates, "
            "not guaranteed or independently calibrated confidence levels."
        )

        st.markdown("### 🚙 Submitted Vehicle Details")

        result_details = pd.DataFrame({
            "Parameter": [
                "Car Name",
                "Vehicle Age",
                "Kilometers Driven",
                "Previous Owners",
                "Seller Type",
                "Transmission"
            ],
            "Value": [
                car_name,
                f"{vehicle_age} years",
                f"{kms_driven:,} km",
                owner,
                seller_type,
                transmission
            ]
        })

        st.dataframe(
            result_details,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "FuelSelect-X provides an experimental machine-learning "
            "prediction. Verify the actual fuel type using official "
            "vehicle documentation."
        )

    except Exception as error:

        st.error(f"Prediction error: {error}")


# -----------------------------------------
# MODEL EXPLAINABILITY DASHBOARD
# -----------------------------------------

st.divider()

st.markdown("## 🔍 Model Explainability")
st.write(
    "Understand which vehicle features contribute "
    "to the fuel classification model."
)

importance_path = BASE_DIR / "data" / "feature_importance.csv"

if importance_path.exists():

    importance_data = pd.read_csv(importance_path)

    importance_data = importance_data.sort_values(
        by="Importance",
        ascending=False
    )

    st.markdown("### 📊 Feature Importance")

    st.bar_chart(
        importance_data.set_index("Feature")["Importance"],
        horizontal=True
    )

    st.markdown("### Feature Importance Details")

    st.dataframe(
        importance_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Positive importance indicates that shuffling a feature "
        "reduced model performance. Negative importance indicates "
        "that shuffling slightly improved performance."
    )

else:
    st.warning(
        "Feature importance file not found. "
        "Run python explainability.py first."
    )

# -----------------------------------------
# MODEL COMPARISON DASHBOARD
# -----------------------------------------

st.divider()

st.markdown("## 🏆 Model Performance Comparison")

comparison_path = (
    BASE_DIR / "data" / "advanced_model_comparison.csv"
)

if comparison_path.exists():

    comparison_data = pd.read_csv(comparison_path)

    st.markdown("### Cross-Validation Balanced Accuracy")

    st.bar_chart(
        comparison_data.set_index("Model")[
            "Mean_Balanced_Accuracy"
        ],
        horizontal=True
    )

    st.markdown("### Test Set Balanced Accuracy")

    st.bar_chart(
        comparison_data.set_index("Model")[
            "Test_Balanced_Accuracy"
        ],
        horizontal=True
    )

    st.markdown("### Complete Model Comparison")

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Model comparison file not found. "
        "Run python advanced_models.py first."
    )

# -----------------------------------------
# RESEARCH FINDINGS
# -----------------------------------------

st.divider()

st.markdown("## 📝 Research Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Best Balanced Accuracy")

    st.metric(
        "Logistic Regression",
        "78.13%"
    )

    st.write(
        "Logistic Regression achieved the highest "
        "balanced accuracy on the final test set."
    )

with col2:
    st.markdown("### Most Important Feature")

    st.metric(
        "Car Name",
        "0.0953"
    )

    st.write(
        "Car Name showed the highest observed "
        "permutation feature importance."
    )

st.caption(
    "Research findings are based on the current dataset "
    "and experimental model evaluation. Further validation "
    "with larger datasets is required."
)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.markdown(
    "<center>🚗 <b>FuelSelect-X</b> | "
    "Machine Learning Based Fuel Classification</center>",
    unsafe_allow_html=True
)
