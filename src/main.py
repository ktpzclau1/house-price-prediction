import pandas as pd
from pathlib import Path

# Find our dataset
file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "house_prices.csv"
)


# Load house price data
df = pd.read_csv(file_path)

# Display the dataset
print("HOUSE PRICE DATA")
print("--------------------")
print(df)

# Count houses
print("\nTotal houses:", len(df))

# Calculate average house price
average_price = df["Price_SEK"].mean()

print("Average house price:", average_price, "SEK")

# Find smallest and largest houses
smallest_house = df["Size_m2"].min()
largest_house = df["Size_m2"].max()

print("Smallest house:", smallest_house, "m2")
print("Largest house:", largest_house, "m2")