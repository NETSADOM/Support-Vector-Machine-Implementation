# SVM From Scratch

A simple implementation of a **Support Vector Machine (SVM)** from scratch using Python and NumPy to understand how the algorithm works.

## Mathematics

### 1. Decision Function

Calculates the score used to classify a sample.

$$
f(x) = w^T x + b
$$

### 2. Hinge Loss

Penalizes incorrect predictions and samples inside the margin.

$$
L_i = \max(0, 1 - y_i f(x_i))
$$

### 3. Objective Function

Minimizes classification loss while controlling model complexity.

$$
J(w,b) = \frac{\lambda}{2}\|w\|^2
+ \frac{1}{n}\sum_{i=1}^{n} L_i
$$

### 4. Gradient Descent

Updates the weights to minimize the objective function.

$$
w \leftarrow w - \eta \nabla_w J
$$

Where \(\eta\) is the learning rate.

## Technologies

- Python
- NumPy
- Jupyter Notebook

## Key Concepts

- Hinge loss
- Regularization
- Maximum margin
- Gradient descent
- Linear classification

## Goal

To understand the mathematics behind SVM by implementing its core mechanism from scratch instead of using a pre-built classifier.
