import pandas as pd
from pathlib import Path

# Find our CSV file
file_path = Path(__file__).parent / "house_prices.csv"

# Load the dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "house_prices.csv")

print("HOUSE PRICE DATA ANALYSIS")
print("-------------------------")

# Show the first five rows
print("\nFIRST 5 ROWS")
print(df.head())

# Show the last five rows
print("\nLAST 5 ROWS")
print(df.tail())

# Check the dataset shape
print("\nDATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Show the column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Check for missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Check column data types
print("\nDATA TYPES")
print(df.dtypes)

# Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

# Average house size
average_size = df["Size_m2"].mean()

# Average house price
average_price = df["Price_SEK"].mean()

# Median house price
median_price = df["Price_SEK"].median()

# Minimum and maximum prices
min_price = df["Price_SEK"].min()
max_price = df["Price_SEK"].max()

print("\nHOUSE PRICE SUMMARY")
print("-------------------------")
print(f"Average size: {average_size:.2f} m2")
print(f"Average price: {average_price:,.0f} SEK")
print(f"Median price: {median_price:,.0f} SEK")
print(f"Lowest price: {min_price:,.0f} SEK")
print(f"Highest price: {max_price:,.0f} SEK")

# Calculate price per square metre
df["Price_per_m2"] = df["Price_SEK"] / df["Size_m2"]

print("\nPRICE PER SQUARE METRE")
print(df[["Size_m2", "Price_SEK", "Price_per_m2"]])

# Average price per square metre
average_price_m2 = df["Price_per_m2"].mean()

print(f"\nAverage price per m2: {average_price_m2:,.0f} SEK")

# Calculate correlation
correlation = df["Size_m2"].corr(df["Price_SEK"])

print("\nCORRELATION ANALYSIS")
print(f"House size vs price: {correlation:.2f}")

# Save the analysed dataset
output_path = BASE_DIR / "data" / "analysed_house_prices.csv"

df.to_csv(output_path, index=False)

print("\nAnalysis complete!")
print("Saved:", output_path.name)
