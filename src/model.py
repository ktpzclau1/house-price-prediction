import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------
# 1. Load the dataset
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "house_prices.csv")

print("HOUSE PRICE PREDICTION")
print("----------------------")

print("\nDataset loaded!")
print("Total houses:", len(df))

# ---------------------------------
# 2. Choose input and output
# ---------------------------------

# X = House size
X = df[["Size_m2"]]

# y = House price
y = df["Price_SEK"]

# ---------------------------------
# 3. Split the dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTRAINING AND TEST DATA")
print("Training houses:", len(X_train))
print("Testing houses:", len(X_test))

# ---------------------------------
# 4. Create the model
# ---------------------------------

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training complete!")

# ---------------------------------
# 5. Predict test house prices
# ---------------------------------

predictions = model.predict(X_test)

print("\nACTUAL VS PREDICTED PRICES")

results = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": predictions
})

print(results.round(2))

# ---------------------------------
# 6. Evaluate the model
# ---------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print(f"Mean Absolute Error: {mae:,.2f} SEK")
print(f"R-squared Score: {r2:.4f}")

# ---------------------------------
# 7. Predict a new house
# ---------------------------------

new_house = pd.DataFrame({
    "Size_m2": [120]
})

predicted_price = model.predict(new_house)

print("\nNEW HOUSE PREDICTION")
print("House size: 120 m2")
print(f"Predicted price: {predicted_price[0]:,.0f} SEK")

# ---------------------------------
# 8. Show what the model learned
# ---------------------------------

print("\nMODEL PARAMETERS")
print(f"Price per additional m2: {model.coef_[0]:,.0f} SEK")
print(f"Starting price (intercept): {model.intercept_:,.0f} SEK")