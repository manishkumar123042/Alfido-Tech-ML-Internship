import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------
# Load Model
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "house_price_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------
# Header
# --------------------------------

st.title("🏠 House Price Prediction")

st.write(
    "Enter the property details below to estimate "
    "the house price using the trained machine learning model."
)

st.divider()


# --------------------------------
# Property Information
# --------------------------------

st.subheader("Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0.0,
        value=3.0,
        step=1.0
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

    sqft_living = st.number_input(
        "Living Area (sqft)",
        min_value=0,
        value=1800,
        step=100
    )

    sqft_lot = st.number_input(
        "Lot Area (sqft)",
        min_value=0,
        value=5000,
        step=500
    )

with col2:
    floors = st.number_input(
        "Floors",
        min_value=1.0,
        value=1.0,
        step=0.5
    )

    waterfront = st.selectbox(
        "Waterfront",
        [0, 1]
    )

    view = st.number_input(
        "View Rating",
        min_value=0,
        value=0,
        step=1
    )

    condition = st.number_input(
        "Condition",
        min_value=1,
        value=3,
        max_value=5,
        step=1
    )

with col3:
    sqft_above = st.number_input(
        "Above Ground Area (sqft)",
        min_value=0,
        value=1800,
        step=100
    )

    sqft_basement = st.number_input(
        "Basement Area (sqft)",
        min_value=0,
        value=0,
        step=100
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )

    yr_renovated = st.number_input(
        "Year Renovated",
        min_value=0,
        max_value=2026,
        value=0,
        step=1
    )


st.divider()


# --------------------------------
# Location Information
# --------------------------------

st.subheader("Location Information")

col1, col2, col3 = st.columns(3)

with col1:
    date = st.date_input(
        "Listing Date"
    )

with col2:
    street = st.text_input(
        "Street",
        value="Example Street"
    )

    city = st.text_input(
        "City",
        value="Seattle"
    )

with col3:
    statezip = st.text_input(
        "State / ZIP",
        value="WA 98101"
    )

    country = st.text_input(
        "Country",
        value="USA"
    )


# --------------------------------
# Automatically Calculated Features
# --------------------------------

house_age = 2026 - yr_built

renovated = 1 if yr_renovated > 0 else 0

total_sqft = sqft_living + sqft_basement


st.info(
    f"House Age: {house_age} years  |  "
    f"Renovated: {'Yes' if renovated else 'No'}  |  "
    f"Total Area: {total_sqft:,} sqft"
)


# --------------------------------
# Prediction
# --------------------------------

if st.button(
    "🏠 Predict House Price",
    use_container_width=True
):

    new_house = pd.DataFrame([{
        "date": str(date),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "street": street,
        "city": city,
        "statezip": statezip,
        "country": country,
        "house_age": house_age,
        "renovated": renovated,
        "total_sqft": total_sqft
    }])

    try:
        prediction_log = model.predict(new_house)[0]

        prediction = np.expm1(prediction_log)

        st.success("Prediction completed successfully!")

        st.metric(
            "Estimated House Price",
            f"${prediction:,.2f}"
        )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.exception(e)


# --------------------------------
# About Section
# --------------------------------

st.divider()

with st.expander("About this project"):
    st.write(
        """
        This application is part of the Alfido Tech Machine Learning
        Internship.

        The underlying model compares Linear Regression, Random Forest,
        and Gradient Boosting for house price prediction.

        The final model includes preprocessing, categorical encoding,
        numerical scaling, and the trained regression model in a
        single pipeline.
        """
    )