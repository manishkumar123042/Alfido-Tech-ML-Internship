import streamlit as st
import pickle
import os
import numpy as np


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Iris Classification",
    page_icon="🌸",
    layout="centered"
)


# --------------------------------
# Load Model
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "iris_best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "iris_scaler.pkl"
)


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


# --------------------------------
# Header
# --------------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below to predict "
    "the iris species using the trained machine learning model."
)


# --------------------------------
# Input Section
# --------------------------------

st.subheader("Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2,
        step=0.1
    )


# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict Iris Species", use_container_width=True):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Species: {prediction}")