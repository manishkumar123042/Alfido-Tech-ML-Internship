# House Price Prediction using Machine Learning

## Alfido Tech Machine Learning Internship — Task 2

## Project Overview

This project develops a machine learning regression system to predict house prices using property characteristics.

The project focuses on exploratory data analysis, feature engineering, missing-value handling, feature transformation, categorical encoding, feature scaling, model comparison, regression evaluation, residual analysis, and model serialization.

Three regression algorithms are trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The best-performing model is selected using RMSE and saved using Python Pickle.

## Dataset

**Source:** Kaggle — House Price Prediction Dataset

The dataset contains information about residential properties and their corresponding prices.

The project uses property characteristics such as:

* Number of bedrooms
* Number of bathrooms
* Living area
* Lot area
* Number of floors
* Waterfront information
* View
* Condition
* Above-ground living area
* Basement area
* Year built
* Renovation information
* Location-related features

The exact columns used by the final model are documented in the notebook.

## Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Missing Value Handling
   ↓
Duplicate Removal
   ↓
Feature Engineering
   ↓
Log Transformation
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ├── Linear Regression
   ├── Random Forest
   └── Gradient Boosting
   ↓
RMSE & MAE Evaluation
   ↓
Residual Analysis
   ↓
Best Model Selection
   ↓
Pickle Model
   ↓
Prediction
```

## Exploratory Data Analysis

The following EDA was performed:

* Dataset structure and information
* Statistical summary
* Missing-value analysis
* Duplicate-value analysis
* House price distribution
* Price outlier analysis
* Numerical feature correlation
* Feature-to-price relationships
* Log-transformed price distribution

Important visualizations are stored in the `screenshots` directory.

## Feature Engineering

Additional features were created from the available property information.

Examples include:

* House age
* Renovation indicator
* Total square footage

House age is derived from the construction year, while the renovation indicator identifies whether a property has been renovated.

Total square footage combines relevant living-area measurements where available.

## Feature Transformation

The target price was log-transformed using:

```python
np.log1p(price)
```

This transformation helps reduce the effect of strong right-skewness in house-price data.

Predictions are converted back to the original price scale using:

```python
np.expm1(prediction)
```

## Preprocessing

Numerical features are processed using:

```text
StandardScaler
```

Categorical features are processed using:

```text
OneHotEncoder
```

The preprocessing is implemented using a Scikit-learn `ColumnTransformer` and included inside each model pipeline.

This ensures that preprocessing is fitted using the training data and applied consistently to the test data and future predictions.

## Machine Learning Models

### 1. Linear Regression

Used as a baseline regression model to establish a simple relationship between the input features and house prices.

### 2. Random Forest Regressor

An ensemble model that combines multiple decision trees to capture nonlinear relationships between house characteristics and prices.

### 3. Gradient Boosting Regressor

An ensemble boosting algorithm that builds models sequentially to improve prediction performance.

## Evaluation Metrics

The models are evaluated using:

### RMSE

Root Mean Squared Error measures the average magnitude of prediction errors while giving greater weight to larger errors.

Lower RMSE indicates better performance.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted prices.

Lower MAE indicates better performance.

## Residual Analysis

Residual analysis was performed for the best-performing model.

The project analyzes:

* Residual distribution
* Actual vs predicted prices
* Residuals vs predicted values
* Residual mean and standard deviation

These plots help identify systematic prediction errors and assess the quality of the regression model.

## Model Selection

The model with the lowest RMSE on the test dataset is selected as the final model.

The actual model performance is available in:

```text
notebook/House_Price_Prediction.ipynb
```

## Saved Model

The final trained pipeline is saved using Pickle:

```text
model/house_price_model.pkl
```

The saved pipeline contains the required preprocessing and trained regression model.

Therefore, a separate scaler file is not required.

## Prediction

A prediction example is included in the notebook.

The standalone prediction script is:

```text
src/predict.py
```

Run it from the project root:

```bash
python src/predict.py
```

The script loads the saved model and predicts a house price from example property information.

## Installation

Make sure Python is installed.

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
notebook/House_Price_Prediction.ipynb
```

Run the notebook cells in order.

## Project Structure

```text
Task-02-House-Price-Prediction/
│
├── data/
│   └── data.csv
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── model/
│   └── house_price_model.pkl
│
├── screenshots/
│   ├── price_distribution.png
│   ├── price_boxplot.png
│   ├── correlation_heatmap.png
│   ├── log_price_distribution.png
│   ├── model_comparison.png
│   ├── mae_comparison.png
│   ├── residual_distribution.png
│   ├── actual_vs_predicted.png
│   └── residual_analysis.png
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
* Handled missing values and duplicate records.
* Created meaningful engineered features.
* Applied log transformation to house prices.
* Encoded categorical features.
* Scaled numerical features.
* Compared Linear Regression, Random Forest, and Gradient Boosting.
* Evaluated models using RMSE and MAE.
* Performed residual analysis.
* Selected the best-performing regression model.
* Saved the complete model pipeline using Pickle.
* Provided an example prediction workflow.

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
**Task:** House Price Prediction
**Task Type:** Supervised Machine Learning — Regression

## Author

**Manish Kumar**

B.Tech — Artificial Intelligence and Machine Learning
