# 🛞 Tyre Degradation Prediction using Machine Learning

A machine learning regression project that predicts **tyre degradation** using simulated tyre and environmental data. The project performs exploratory data analysis, feature engineering, preprocessing, model comparison, and prediction using multiple regression algorithms.

The goal is to understand how factors such as tyre wear, humidity, ambient temperature, and event conditions influence tyre degradation.

---

## 📌 Project Overview

Tyre degradation is an important factor in understanding tyre performance and remaining usability.

This project uses a simulated tyre telemetry dataset and evaluates multiple machine learning regression models to predict tyre degradation.

The project compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Multi-Layer Perceptron (MLP)

The best-performing model is selected using cross-validation and evaluated on a held-out test dataset.

---

## 🎯 Objectives

- Perform exploratory data analysis on tyre-related data
- Identify relationships between tyre and environmental conditions
- Clean and preprocess the dataset
- Engineer relevant features
- Compare multiple regression algorithms
- Evaluate models using R² and RMSE
- Select the best-performing model
- Predict tyre degradation for new input data

---

## 📊 Dataset

The project uses the **Simple Tire Wear and Degradation Simulated Dataset** available on Kaggle.

Dataset:

https://www.kaggle.com/datasets/samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset

The dataset contains approximately **2.6 million simulated telemetry records** with around 30 features related to:

- Tyre wear
- Lap information
- Driving conditions
- Weather
- Track conditions
- Tyre compound
- Environmental conditions

For this project, four features are used for modelling:

| Feature | Description |
|---|---|
| `Tire_wear` | Current tyre wear level |
| `Humidity` | Environmental humidity |
| `Ambient_Temperature` | Ambient temperature |
| `Event` | Event/track condition |

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand the dataset and identify relationships between variables.

The analysis includes:

- Dataset overview
- Missing-value analysis
- Statistical summaries
- Feature distributions
- Correlation analysis
- Outlier analysis
- Relationship between tyre wear and degradation

Visualization is performed using:

- Matplotlib
- Seaborn

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Loading the raw CSV dataset
2. Cleaning column names
3. Selecting relevant features
4. Separating features and target
5. Handling numerical features
6. Standardizing numerical features
7. One-hot encoding categorical features
8. Splitting the data into training and testing sets

All preprocessing steps are implemented inside an `sklearn` Pipeline to prevent data leakage.

---

## 🤖 Machine Learning Models

Four regression algorithms are compared:

### 1. Linear Regression

Used as a baseline regression model.

### 2. Decision Tree Regressor

Captures non-linear relationships between tyre and environmental conditions.

### 3. Random Forest Regressor

An ensemble model that combines multiple decision trees to improve prediction performance.

### 4. Multi-Layer Perceptron

A neural-network-based regression model used to compare traditional ML with a neural approach.

---

## 📏 Model Evaluation

The models are evaluated using:

### R² Score

Measures how much variance in the target variable is explained by the model.

Higher is better.

### RMSE

Measures the average prediction error in the target's units.

Lower is better.

---

## 📈 Model Results

Initial model comparison:

| Model | R² | RMSE |
|---|---:|---:|
| Linear Regression | 0.83 | 15.33 |
| Decision Tree | 0.76 | 18.46 |
| **Random Forest** | **0.88** | **13.10** |
| MLP | 0.88 | 13.14 |

Based on the initial evaluation, **Random Forest** provided the best overall performance.

> Note: Results may vary slightly depending on the train/test split and model configuration.

---

## 🔄 Cross-Validation

Instead of relying only on a single train/test split, the project uses **5-fold cross-validation** for model selection.

The workflow is:

```text
Dataset
   ↓
Train / Test Split
   ↓
5-Fold Cross Validation
   ↓
Compare Regression Models
   ↓
Select Best Model
   ↓
Train Best Model
   ↓
Evaluate on Held-Out Test Set
