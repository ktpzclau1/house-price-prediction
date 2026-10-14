import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# 1. Find project files
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(exist_ok=True)

# ---------------------------------
# 2. Load the dataset
# ---------------------------------

df = pd.read_csv(BASE_DIR / "data" / "house_prices.csv")

X = df[["Size_m2"]]
y = df["Price_SEK"]

# ---------------------------------
# 3. Split training and testing data
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ---------------------------------
# 4. Train the model
# ---------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices for test data
predictions = model.predict(X_test)

# ---------------------------------
# 5. Calculate evaluation metrics
# ---------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MODEL EVALUATION")
print("-------------------------")

print(f"Mean Absolute Error: {mae:,.2f} SEK")
print(f"Root Mean Squared Error: {rmse:,.2f} SEK")
print(f"R-squared Score: {r2:.4f}")

# ---------------------------------
# 6. Compare actual and predicted
# ---------------------------------

results = pd.DataFrame({
    "Size_m2": X_test["Size_m2"].to_numpy(),
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": predictions
})

results["Error"] = (
    results["Actual_Price"] - results["Predicted_Price"]
)

results = results.sort_values("Size_m2")

print("\nACTUAL VS PREDICTED PRICES")
print(results.round(2).to_string(index=False))

# ---------------------------------
# 7. Save predictions to CSV
# ---------------------------------

output_path = BASE_DIR / "model_predictions.csv"

results.to_csv(output_path, index=False)

print("\nSaved:", output_path.name)

# ---------------------------------
# 8. Chart 1: Actual vs Predicted
# ---------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    results["Actual_Price"],
    results["Predicted_Price"],
    color="royalblue",
    s=80,
    label="Model predictions"
)

# Perfect prediction reference line
min_price = min(
    results["Actual_Price"].min(),
    results["Predicted_Price"].min()
)

max_price = max(
    results["Actual_Price"].max(),
    results["Predicted_Price"].max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (SEK)")
plt.ylabel("Predicted Price (SEK)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ---------------------------------
# 9. Chart 2: Prediction Errors
# ---------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    results["Size_m2"],
    results["Error"],
    color="mediumpurple",
    s=80
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Zero error"
)

plt.title("Prediction Errors by House Size")
plt.xlabel("House Size (m²)")
plt.ylabel("Error (SEK)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "prediction_errors.png",
    dpi=300
)

plt.show()

print("\nEvaluation complete!")
print("Charts saved in:", IMAGE_DIR)