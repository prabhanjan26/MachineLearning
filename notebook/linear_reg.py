import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config.configuration import DATA_PATH


# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv(DATA_PATH)

# print(df.head())
# print(df.info())


# ---------------------------
# Features & Target
# ---------------------------

X = df.drop(columns=["Price_Lakhs"])
y = df["Price_Lakhs"]


# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------
# Model
# ---------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------
# Model Parameters
# ---------------------------

print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:<20} {coef:.4f}")


# ---------------------------
# Prediction
# ---------------------------

y_pred = model.predict(X_test)


# ---------------------------
# Evaluation
# ---------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ---------------------------
# Predict New House
# ---------------------------

new_house = pd.DataFrame({
    "Area_sqft": [1800],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age_Years": [5],
    "Distance_City_km": [7.5]
})

predicted_price = model.predict(new_house)

print(f"\nPredicted Price: ₹{predicted_price[0]:.2f} Lakhs")


# ---------------------------
# Actual vs Predicted
# ---------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\n")
print(results.head(10))


# ---------------------------
# Visualization
# ---------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()