
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_car_data.csv")

# Reference year for vehicle age
CURRENT_YEAR = 2026

# Feature 1: Vehicle Age
df["Vehicle_Age"] = CURRENT_YEAR - df["Year"]

# Feature 2: Price Difference
df["Price_Difference"] = (
    df["Present_Price"] - df["Selling_Price"]
)

# Feature 3: Depreciation Ratio
df["Depreciation_Ratio"] = (
    df["Selling_Price"] / df["Present_Price"]
)

# Display new features
print("\nNEW FEATURES CREATED")

print(df[[
    "Car_Name",
    "Year",
    "Vehicle_Age",
    "Price_Difference",
    "Depreciation_Ratio",
    "Fuel_Type"
]].head(10))

# Check feature statistics
print("\nFEATURE STATISTICS")

print(df[[
    "Vehicle_Age",
    "Price_Difference",
    "Depreciation_Ratio"
]].describe())

# Save engineered dataset
df.to_csv(
    "data/engineered_car_data.csv",
    index=False
)

print("\nFEATURE ENGINEERING COMPLETED!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Saved: data/engineered_car_data.csv")
