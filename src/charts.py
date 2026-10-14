import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find our project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Create the images folder if it doesn't exist
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

# Load the house price dataset
df = pd.read_csv(BASE_DIR / "data" / "house_prices.csv")

# ---------------------------------
# CHART 1: House Size vs. Price
# ---------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    df["Size_m2"],
    df["Price_SEK"],
    color="royalblue",
    s=80
)

plt.title("House Size vs. House Price")
plt.xlabel("House Size (m²)")
plt.ylabel("House Price (SEK)")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(IMAGE_DIR / "house_size_vs_price.png", dpi=300)
plt.show()

# ---------------------------------
# CHART 2: House Price Distribution
# ---------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["Price_SEK"],
    bins=6,
    color="mediumpurple",
    edgecolor="black"
)

plt.title("Distribution of House Prices")
plt.xlabel("House Price (SEK)")
plt.ylabel("Number of Houses")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(IMAGE_DIR / "price_distribution.png", dpi=300)
plt.show()

# ---------------------------------
# CHART 3: House Price Trend
# ---------------------------------

# Sort by size before drawing the line
sorted_df = df.sort_values("Size_m2")

plt.figure(figsize=(9, 5))

plt.plot(
    sorted_df["Size_m2"],
    sorted_df["Price_SEK"],
    color="seagreen",
    marker="o",
    linewidth=2
)

plt.title("House Price Trend by Size")
plt.xlabel("House Size (m²)")
plt.ylabel("House Price (SEK)")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(IMAGE_DIR / "house_price_trend.png", dpi=300)
plt.show()

print("All charts created successfully!")
print("Saved in:", IMAGE_DIR)