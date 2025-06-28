import numpy as np
import pandas as pd
from LinearRegression import LinearRegression

class LassoRegression(LinearRegression) :
    def __init__ (self, learning_rate = 0.1, iterations = 100, verbose = True, lambda_ = 0.1):
        super().__init__(learning_rate, iterations, verbose)
        self.lambda_ = lambda_

    def compute_loss(self,Y_pred,Y):
        return np.mean(np.sum((Y - Y_pred) ** 2, axis=1)) + np.sum(np.abs(self.W))

    def compute_grad(self,X,Y_pred,Y):
        n = X.shape[0]
        grad_w = (2/n)*((X.T) @ (Y_pred - Y)) + self.lambda_*np.sign(self.W)
        grad_b = (2/n)*np.sum(Y_pred - Y,axis = 0)
        return grad_w, grad_b