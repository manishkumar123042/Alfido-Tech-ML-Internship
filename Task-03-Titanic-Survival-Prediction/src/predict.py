import os
import pickle
import pandas as pd


# --------------------------------
# Project directory
# --------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "titanic_survival_model.pkl"
)


# --------------------------------
# Load saved model
# --------------------------------

if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found!")
    print("Expected location:", MODEL_PATH)
    exit()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully.")


# --------------------------------
# Example passenger
# --------------------------------

passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.05,
    "Embarked": "S",
    "Title": "Mr",
    "FamilySize": 1,
    "CabinPresent": 0
}])


# --------------------------------
# Make prediction
# --------------------------------

prediction = model.predict(passenger)[0]


# --------------------------------
# Display result
# --------------------------------

print()
print("Titanic Survival Prediction")
print("---------------------------")

if prediction == 1:
    print("Prediction: Passenger is likely to survive.")
else:
    print("Prediction: Passenger is unlikely to survive.")