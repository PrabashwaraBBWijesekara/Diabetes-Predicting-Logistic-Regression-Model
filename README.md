# Diabetes-Predicting-Logistic-Regression-Model

# MA4144 In-Class Project 1: Basics

## Overview

Train a logistic regression model on the diabetes dataset to predict the risk of diabetes given patient data then use it to perform predictions on previously unseen (test) data.

## Features

The project includes the following tasks:

1. In the first section  it will be analysed a dataset to look into its statistical propeties. For this we will use the DiabetesTrain.csv dataset. Each row corresponds to a single patient. The first 8 columns correspond to the features of the patients that may help predict risk of diabetes. The outcome
column is a binary column represting the risk of diabetes, outcome 1 : high risk of diabetes and
outcome 0 little to no risk of diabetes.

2. In the second section it will be trained a logistic regression model on the diabetes dataset to predict the risk of diabetes given patient data then use it to perform predictions on previously unseen (test) data.

## Model Architecture
## Model Architecture

This project uses a **logistic regression** model for binary classification. The architecture is composed of the following components:

### 1. Input Layer
- Accepts a feature vector `x ∈ ℝⁿ`, where `n` is the number of normalized input features (after preprocessing, typically 8 for the diabetes dataset).
- Each input is normalized using the training dataset’s mean and standard deviation.

### 2. Linear Transformation
- The model computes a weighted sum of inputs:  
  \[
  z = \mathbf{w}^T \mathbf{x} + b
  \]
  where:
  - `w ∈ ℝⁿ`: weight vector
  - `b ∈ ℝ`: bias (scalar)
  - `z ∈ ℝ`: raw score

### 3. Activation Function
- Applies the sigmoid function to the linear output:
  \[
  \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
  - Outputs a probability score `ŷ ∈ (0, 1)` indicating the likelihood of the positive class.

### 4. Prediction
- Converts the probability into a binary class label:
  \[
  \hat{y}_{label} =
  \begin{cases}
  1 & \text{if } \hat{y} \geq 0.5 \\
  0 & \text{otherwise}
  \end{cases}
  \]

### 5. Loss Function
- Uses **Binary Cross-Entropy Loss**:
  \[
  \mathcal{L} = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
  - Optionally includes regularization:
    - **L2 (Ridge)**: adds `λ * ||w||²` penalty
    - **L1 (Lasso)**: adds `λ * ||w||₁` penalty

### 6. Optimization
- Trained using **Gradient Descent**:
  - Updates weights and bias iteratively using computed gradients:
    \[
    w := w - \eta \cdot \frac{∂\mathcal{L}}{∂w} \\
    b := b - \eta \cdot \frac{∂\mathcal{L}}{∂b}
    \]
  - `η`: learning rate
  - Gradients are derived analytically from the loss function, with or without regularization.

## Results

- The model achieves a classification accuracy of **76.19%** on training data, which is a good baseline performance for a logistic regression model on this dataset.
- Visualization of the loss shows a smooth convergence over iterations.
- Weight norm plot confirms stability of the weight updates during training.
- Cross-validation ensures generalizability by selecting optimal hyperparameters before final evaluation.

This project showcases a complete pipeline for binary classification using logistic regression, emphasizing both mathematical understanding and implementation efficiency using NumPy.

