
import pandas as pd

# Load dataset
df = pd.read_csv("data/car_data.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nFUEL TYPE DISTRIBUTION")
print(df["Fuel_Type"].value_counts())
