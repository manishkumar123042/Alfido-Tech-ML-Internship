# Iris Classification using Machine Learning

## Alfido Tech Machine Learning Internship — Task 01

## Project Overview

This project implements a machine learning classification system to predict the species of an iris flower using its sepal and petal measurements.

The project covers exploratory data analysis, data preprocessing, supervised machine learning, model comparison, evaluation, model serialization, and example inference.

## Dataset

The project uses the Iris Classification Dataset provided for the Alfido Tech internship task.

**Dataset Source:** Kaggle — Iris Classification Dataset

Features used for prediction:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Target:

* Iris Species

## Machine Learning Algorithms

Three classification algorithms were trained and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree

## Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ├── Logistic Regression
   ├── KNN
   └── Decision Tree
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Pickle Model
   ↓
Example Prediction
```

## Exploratory Data Analysis

The following analysis was performed:

* Dataset structure and information
* Statistical summary
* Missing-value analysis
* Duplicate-value analysis
* Species distribution
* Feature relationship visualization
* Correlation analysis

Important visualizations are available in the `screenshots` folder.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix
* Classification Report

The final model comparison is available in the notebook.

> The performance values shown in the final report are based on the actual results obtained during model training and testing.

## Model Saving

The best-performing model is saved using Python's `pickle` module:

```text
model/iris_best_model.pkl
```

The feature scaler is also saved:

```text
model/iris_scaler.pkl
```

## Example Inference

A new iris flower can be classified using its four measurements.

Example:

```python
new_flower = [[5.1, 3.5, 1.4, 0.2]]
```

The prediction can be generated using the saved model.

The standalone prediction script is available at:

```text
src/predict.py
```

Run it from the project root:

```bash
python src/predict.py
```

## Installation

Make sure Python is installed on your system.

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Required Python Packages

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebook/Iris_Classification.ipynb
```

You can also open the notebook directly in VS Code.

## Project Structure

```text
Task-01-Iris-Classification/
│
├── data/
│   └── Iris.csv
│
├── notebook/
│   └── Iris_Classification.ipynb
│
├── model/
│   ├── iris_best_model.pkl
│   └── iris_scaler.pkl
│
├── screenshots/
│   ├── class_distribution.png
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
├── src/
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Outcomes

* Performed exploratory data analysis on the Iris dataset.
* Visualized the separability of iris species.
* Trained three supervised classification algorithms.
* Compared model performance using multiple evaluation metrics.
* Selected the best-performing model based on test performance.
* Saved the trained model using Pickle.
* Implemented an example inference workflow.

## Internship Task

**Organization:** Alfido Tech
**Task:** Machine Learning — Iris Classification
**Task Type:** Supervised Machine Learning Classification

## Author

**Manish Kumar**

B.Tech — Artificial Intelligence and Machine Learning
