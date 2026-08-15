import os
import pickle


# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model and scaler paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "iris_scaler.pkl")


# Check whether model exists
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found!")
    print("Expected location:", MODEL_PATH)
    exit()


# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Load the scaler
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
else:
    scaler = None


# New iris flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]


# Make prediction
if scaler is not None:
    try:
        new_flower_scaled = scaler.transform(new_flower)
        prediction = model.predict(new_flower_scaled)
    except:
        prediction = model.predict(new_flower)
else:
    prediction = model.predict(new_flower)


print("Predicted Iris Species:", prediction[0])