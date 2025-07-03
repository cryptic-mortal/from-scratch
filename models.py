import numpy as np
import pandas as pd

class BinaryLogisticRegression():
    def __init__(self, learning_rate = 0.1, iterations = 100, verbose = False, threshold = 0.5) :
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []
        self.w = None
        self.b = None
        self.w_history = []
        self.b_history = []
        self.threshold = threshold
    @staticmethod
    def sigmoid(z) :
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))

    def fit(self, X, y) :
        self.n, self.m = X.shape
        self.w = np.zeros((self.m,1))
        self.b = 0
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for i in range(self.iterations) :
            y_pred = self.predict_raw(X)
            
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)

            dJ_dw, dJ_db = self.compute_grad(X, y_pred, y)
            self.w -= self.learning_rate*dJ_dw
            self.b -= self.learning_rate*dJ_db
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
            if self.verbose and i % 50 == 0:
                print(f"Iteration {i}, Loss: {loss}")
            
    def predict_raw(self, X) :
        Z = X @ self.w + self.b
        y_pred = self.sigmoid(Z)
        return y_pred

    def predict(self, X) :
        Z = X @ self.w + self.b
        y_pred = self.sigmoid(Z)
        y_pred = (y_pred >= self.threshold)
        return y_pred
        
    def compute_loss(self, y_pred, y) :
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = y*np.log(y_pred) + (1-y)*np.log(1-y_pred)
        cost = -np.mean(loss)
        return cost

    def compute_grad(self, X, y_pred, y) :
        dJ_dw = (X.T @ (y_pred - y))/self.n
        dJ_db = np.mean(y_pred - y)
        return dJ_dw, dJ_db

    def accuracy(self, y_pred, y) :
        return np.mean(y_pred.flatten()==y.flatten())

class LinearRegression():
    def __init__ (self, learning_rate = 0.1, iterations = 100, verbose = False):
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

class LassoRegression(LinearRegression) :
    def __init__ (self, learning_rate = 0.1, iterations = 100, verbose = False, lambda_ = 0.1):
        super().__init__(learning_rate, iterations, verbose)
        self.lambda_ = lambda_

    def compute_loss(self,Y_pred,Y):
        return np.mean(np.sum((Y - Y_pred) ** 2, axis=1)) + np.sum(np.abs(self.W))

    def compute_grad(self,X,Y_pred,Y):
        n = X.shape[0]
        grad_w = (2/n)*((X.T) @ (Y_pred - Y)) + self.lambda_*np.sign(self.W)
        grad_b = (2/n)*np.sum(Y_pred - Y,axis = 0)
        return grad_w, grad_b

class RidgeRegression(LinearRegression) :
    def __init__ (self, learning_rate = 0.1, iterations = 1000, verbose = False, lambda_ = 0.1):
        super().__init__(learning_rate, iterations, verbose)
        self.lambda_ = lambda_

    def compute_loss(self,Y_pred,Y):
        return np.mean(np.sum((Y - Y_pred) ** 2, axis=1)) + np.sum(self.W**2)

    def compute_grad(self,X,Y_pred,Y):
        n = X.shape[0]
        grad_w = (2/n)*((X.T) @ (Y_pred - Y)) + self.lambda_*2*self.W
        grad_b = (2/n)*np.sum(Y_pred - Y,axis = 0)
        return grad_w, grad_b