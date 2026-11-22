
import pandas as pd

# Load original dataset
df = pd.read_csv("data/car_data.csv")

print("ORIGINAL DATASET SHAPE:", df.shape)

# Check duplicate records
print("\nDuplicate records before cleaning:")
print(df.duplicated().sum())

# Display duplicate records
print("\nDuplicate rows:")
print(df[df.duplicated(keep=False)])

# Remove exact duplicate records
df = df.drop_duplicates()

# Check missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Check fuel distribution
print("\nFuel distribution after cleaning:")
print(df["Fuel_Type"].value_counts())

# Calculate class percentages
print("\nFuel type percentages:")
print((df["Fuel_Type"].value_counts(normalize=True) * 100).round(2))

# Save cleaned dataset
df.to_csv("data/cleaned_car_data.csv", index=False)

print("\nCLEANING COMPLETED!")
print("Original records: 301")
print("Records after cleaning:", len(df))
print("Cleaned dataset saved successfully!")
