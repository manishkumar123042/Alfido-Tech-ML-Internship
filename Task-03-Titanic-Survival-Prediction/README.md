# Titanic Survival Prediction using Machine Learning

## Alfido Tech Machine Learning Internship — Task 3

## Project Overview

This project develops a machine learning classification system to predict whether a passenger survived the Titanic disaster using passenger information.

The project focuses on feature engineering, missing-value handling, categorical encoding, model training, evaluation, and model explainability.

The following classification algorithms are trained and compared:

1. Logistic Regression
2. Random Forest Classifier

The best-performing model is selected based on classification performance and saved using Python Pickle.

## Dataset

**Source:** Kaggle — Titanic Survival Dataset

The dataset contains passenger information and the target variable `Survived`.

The project uses passenger attributes including:

* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Port of embarkation
* Passenger title
* Family size
* Cabin presence

## Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ├── Passenger Title
   ├── Family Size
   └── Cabin Presence
   ↓
Missing Value Handling
   ├── Age
   ├── Cabin
   └── Embarked
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ├── Logistic Regression
   └── Random Forest
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Best Model Selection
   ↓
Pickle Model
   ↓
Inference
```

## Exploratory Data Analysis

The notebook performs exploratory analysis of:

* Overall survival distribution
* Survival by gender
* Survival by passenger class
* Passenger age distribution
* Age distribution by survival status

The generated visualizations are stored in the `screenshots` directory.

## Feature Engineering

Three important features were created as required by the internship task.

### Passenger Title

Titles such as `Mr`, `Mrs`, `Miss`, and other titles are extracted from the passenger's name.

Rare titles are grouped into a common `Rare` category.

### Family Size

Family size is calculated using:

```python
FamilySize = SibSp + Parch + 1
```

The additional `1` represents the passenger.

### Cabin Presence

A binary feature is created:

```text
0 → Cabin information unavailable
1 → Cabin information available
```

This allows the model to use cabin availability without relying directly on individual cabin numbers.

## Missing Value Handling

### Age

Missing age values are replaced using the median age.

### Cabin

Missing cabin information is represented as `Unknown`, while the `CabinPresent` feature indicates whether cabin information was originally available.

### Embarked

Missing embarkation values are replaced using the most frequent category.

## Preprocessing

Numerical features are standardized using:

```text
StandardScaler
```

Categorical features are encoded using:

```text
OneHotEncoder
```

The preprocessing is implemented using a Scikit-learn `ColumnTransformer` and included inside the model pipelines.

## Machine Learning Models

### Logistic Regression

Used as a simple and interpretable classification baseline.

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees and can capture nonlinear relationships between passenger characteristics and survival.

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The actual performance values are available in the notebook.

## Model Explainability

Feature importance is used to explain which processed passenger attributes contribute most strongly to the model's predictions.

For Random Forest, the model's built-in feature importance values are used.

For Logistic Regression, the absolute model coefficients are used when it is selected as the best model.

The feature-importance visualization is saved in:

```text
screenshots/feature_importance.png
```

## Saved Model

The best-performing complete pipeline is saved using Python Pickle:

```text
model/titanic_survival_model.pkl
```

The saved pipeline contains the preprocessing steps and trained classification model.

## Inference

A standalone prediction script is provided:

```text
src/predict.py
```

Run it from the project root:

```bash
python src/predict.py
```

The script loads the saved model and predicts whether an example passenger is likely to survive.

Example input:

```python
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
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebook/Titanic_Survival_Prediction.ipynb
```

Run the notebook cells in order.

## Project Structure

```text
Task-03-Titanic-Survival-Prediction/
│
├── data/
│   └── Titanic.csv
│
├── notebook/
│   └── Titanic_Survival_Prediction.ipynb
│
├── model/
│   └── titanic_survival_model.pkl
│
├── screenshots/
│   ├── survival_distribution.png
│   ├── survival_by_gender.png
│   ├── survival_by_class.png
│   ├── age_distribution.png
│   ├── age_survival.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── src/
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Outcomes

* Performed exploratory data analysis.
* Created passenger title features.
* Created family-size features.
* Created cabin-presence features.
* Handled missing Age, Cabin, and Embarked values.
* Encoded categorical variables.
* Scaled numerical variables.
* Compared Logistic Regression and Random Forest.
* Evaluated classification performance using multiple metrics.
* Performed model explainability using feature importance.
* Saved the complete best-performing model using Pickle.
* Provided a standalone inference script.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Jupyter Notebook

## Internship Task

**Organization:** Alfido Tech
**Domain:** Machine Learning
**Task:** Titanic Survival Prediction
**Task Type:** Supervised Machine Learning — Classification

## Author

**Manish Kumar**

B.Tech — Artificial Intelligence and Machine Learning
