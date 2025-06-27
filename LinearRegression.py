import numpy as np
import pandas as pd

class LinearRegression():
    def __init__ (self, learning_rate = 0.1, iterations = 100, verbose = True):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []
        self.W = None
        self.W_history = []
        self.b_history = []
    
    def fit(self,X,Y):
        self.n, self.m = X.shape
        self.k = Y.shape[1]
        self.W = np.zeros((self.m,self.k))
        self.b = np.zeros(self.k)

        for i in range(self.iterations) :
            Y_pred = self.predict(X)
            cost = self.compute_loss(Y_pred,Y)
            self.loss_history.append(cost)
            grad_w, grad_b = self.compute_grad(X,Y_pred,Y)
            self.W = self.W - self.learning_rate*(grad_w)
            self.b = self.b - self.learning_rate*(grad_b)
            self.W_history.append(self.W.copy())
            self.b_history.append(self.b.copy())
            
            if self.verbose and i % 50 == 0:
                print(f"Iteration {i}, Loss: {cost}")
    def predict(self, X):
        return (X @ self.W) + self.b
        
    def compute_loss(self,Y_pred,Y):
        return np.mean(np.sum((Y - Y_pred) ** 2, axis=1))
    def compute_grad(self,X,Y_pred,Y):
        n = X.shape[0]
        grad_w = (2/n)*((X.T) @ (Y_pred - Y))
        grad_b = (2/n)*np.sum(Y_pred - Y,axis = 0)
        return grad_w, grad_b
    