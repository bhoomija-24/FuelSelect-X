
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_car_data.csv")

# Create folder for graphs
os.makedirs("eda_results", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Fuel Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Fuel_Type")
plt.title("Distribution of Vehicle Fuel Types")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Vehicles")
plt.savefig("eda_results/fuel_distribution.png")
plt.show()

# 2. Selling Price by Fuel Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price Distribution by Fuel Type")
plt.savefig("eda_results/price_by_fuel.png")
plt.show()

# 3. Present Price by Fuel Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Present_Price")
plt.title("Present Price by Fuel Type")
plt.savefig("eda_results/present_price_by_fuel.png")
plt.show()

# 4. Kilometers Driven by Fuel Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Kms_Driven")
plt.title("Kilometers Driven by Fuel Type")
plt.savefig("eda_results/km_by_fuel.png")
plt.show()

# 5. Vehicle Year Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Year", hue="Fuel_Type",
             multiple="stack", bins=15)
plt.title("Vehicle Manufacturing Year Distribution")
plt.savefig("eda_results/year_distribution.png")
plt.show()

# 6. Numerical Feature Correlation
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))
sns.heatmap(numeric_df.corr(), annot=True,
            cmap="coolwarm", fmt=".2f")
plt.title("Numerical Feature Correlation")
plt.savefig("eda_results/correlation_heatmap.png")
plt.show()

# 7. Fuel Type and Seller Type
print("\nFUEL TYPE VS SELLER TYPE")
print(pd.crosstab(df["Fuel_Type"], df["Seller_Type"]))

# 8. Fuel Type and Transmission
print("\nFUEL TYPE VS TRANSMISSION")
print(pd.crosstab(df["Fuel_Type"], df["Transmission"]))

# 9. Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All graphs saved inside eda_results folder.")
