import os
import pickle
import pandas as pd
import numpy as np


# --------------------------------
# Project directory
# --------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "house_price_model.pkl"
)


# --------------------------------
# Load model
# --------------------------------

if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found!")
    print("Expected location:", MODEL_PATH)
    exit()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully.")


# --------------------------------
# New house data
# --------------------------------

new_house = pd.DataFrame([{
    "date": "2014-05-02 00:00:00",
    "bedrooms": 3.0,
    "bathrooms": 1.5,
    "sqft_living": 1340,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1340,
    "sqft_basement": 0,
    "yr_built": 1953,
    "yr_renovated": 0,
    "street": "Example Street",
    "city": "Seattle",
    "statezip": "WA 98101",
    "country": "USA",
    "house_age": 73,
    "renovated": 0,
    "total_sqft": 1340
}])


# --------------------------------
# Prediction
# --------------------------------

prediction_log = model.predict(new_house)[0]

prediction = np.expm1(prediction_log)

print()
print("House Prediction")
print("----------------")
print(f"Predicted House Price: ${prediction:,.2f}")