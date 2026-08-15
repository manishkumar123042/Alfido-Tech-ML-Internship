import streamlit as st
import pandas as pd
import pickle
import os


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)


# --------------------------------
# Load Model
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "titanic_survival_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------
# Header
# --------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger information to predict whether "
    "the passenger is likely to survive."
)

st.divider()


# --------------------------------
# Passenger Information
# --------------------------------

st.subheader("Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

with col2:

    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=8.05,
        step=1.0
    )

with col3:

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )

    title = st.selectbox(
        "Passenger Title",
        [
            "Mr",
            "Miss",
            "Mrs",
            "Master",
            "Rare"
        ]
    )

    cabin_present = st.selectbox(
        "Cabin Information Available?",
        ["No", "Yes"]
    )


# --------------------------------
# Automatically Calculate Family Size
# --------------------------------

family_size = sibsp + parch + 1

cabin_value = 1 if cabin_present == "Yes" else 0


st.info(
    f"Family Size: {family_size} | "
    f"Cabin Present: {'Yes' if cabin_value else 'No'}"
)


# --------------------------------
# Prediction
# --------------------------------

st.divider()

if st.button(
    "🚢 Predict Survival",
    use_container_width=True
):

    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title,
        "FamilySize": family_size,
        "CabinPresent": cabin_value
    }])

    try:

        prediction = model.predict(passenger)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success(
                "🟢 Passenger is likely to survive."
            )

        else:

            st.error(
                "🔴 Passenger is unlikely to survive."
            )

        # --------------------------------
        # Prediction Probability
        # --------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(passenger)[0]

            survival_probability = probabilities[1]

            st.metric(
                "Estimated Survival Probability",
                f"{survival_probability * 100:.2f}%"
            )

    except Exception as e:

        st.error("Prediction could not be completed.")

        st.exception(e)


# --------------------------------
# About
# --------------------------------

st.divider()

with st.expander("About this project"):

    st.write(
        """
        This application is part of the Alfido Tech Machine Learning
        Internship.

        The model predicts Titanic passenger survival using passenger
        class, gender, age, family information, fare, embarkation,
        passenger title, family size, and cabin availability.

        Feature engineering and preprocessing were performed during
        model training.
        """
    )