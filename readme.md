<h1 align=center>Models from Scratch</h1>

### Author: [Manthan](https://github.com/cryptic-mortal)  
### Language: Python 
## Table of Contents
- [Overview](#overview)
- [Linear Regression](#linear-regression)
  - [Class Structure](#class-structure)
  - [Example Usage](#example-usage)
- [Installation](#installation)
- [To Do](#to-do)
## Overview
This project is a from-scratch implementation of Machine Learning and Deep Learning models. The goal is to create models from scratch to gain a clear understanding of how these models work under the hood, without relying on high-level libraries like Sklearn, TensorFlow or PyTorch.

- [Linear Regression](#linear-regression)
- [Lasso Regression](#lasso-regression)
## Models
### Linear Regression
This implementation of [Linear Regression](LinearRegression.py) supports:
- Multi-Target regression (`Y.shape = (n, k)`)
- Mean Squared Error (MSE) loss
- Batch Gradient Descent
- Tracks:
  - `loss_history`
  - `W_history` (weights over time)
  - `b_history` (biases over time)
- Optional logging (`verbose=True`)

<br>
Class Structure:

```python
class LinearRegression():
    def __init__(self, learning_rate=0.1, iterations=100, verbose=True)
    def fit(self, X, Y)
    def predict(self, X)
    def compute_loss(self, Y_pred, Y)
    def compute_grad(self, X, Y_pred, Y)
```

Example usage:

```python
from LinearRegression import LinearRegression
import numpy as np 

my_model = LinearRegression(learning_rate = 0.03, iterations = 1000)
my_model.fit(X_train,y_train)
y_pred = my_model.predict(X_test)
print(f"Loss : {my_model.compute_loss(y_pred,y_test)}")
```
For more details, refer to the [Linear Regression Implementation](LR.ipynb).

### Lasso Regression
This implementation of [Lasso Regression](LassoRegression.py) supports:
- Multi-Target regression (`Y.shape = (n, k)`)
- Mean Squared Error (MSE) loss
- L1 regularization
- Batch Gradient Descent
- Tracks:
  - `loss_history`
  - `W_history` (weights over time)
  - `b_history` (biases over time)
- Optional logging (`verbose=True`)

Lasso Regression is a regularized version of Linear Regression that adds an L1 penalty to the loss function, which can help in feature selection by driving some weights to zero.

Class Structure:

```python
class LassoRegression(LinearRegression):
    def __init__(self, learning_rate=0.1, iterations=100, verbose=True, lambda_=0.1)
    def compute_loss(self, Y_pred, Y)
    def compute_grad(self, X, Y_pred, Y)
```
Example usage:

```python
from LassoRegression import LassoRegression
import numpy as np

my_model = LassoRegression(learning_rate = 0.03, iterations = 1000, lambda_=0.1)
my_model.fit(X_train, y_train)
y_pred = my_model.predict(X_test)
print(f"Loss : {my_model.compute_loss(y_pred, y_test)}")
```
For more details, refer to the [Lasso Regression Implementation](Lasso.ipynb).
## Installation
To use this project, clone the repository:
```bash
git clone https://github.com/cryptic-mortal/Models-from-Scratch.git
```
Then, you can use the models by importing them into your Python scripts or Jupyter notebooks.

## To Do
- [] Implement more models such as:
  - [] Logistic Regression

## Credits
- [Machine Learning Course](https://www.youtube.com/playlist?list=PLfFghEzKVmjsNtIRwErklMAN8nJmebB0I) by Siddhardhan
- [Lasso Regression](https://www.geeksforgeeks.org/machine-learning/what-is-lasso-regression/) by GeeksforGeeks