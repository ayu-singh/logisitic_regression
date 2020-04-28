import numpy as np

class LogisiticRegression:

    def __init__(self, lr=0.001, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        # initializing weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #implementing gradient descent
        for i_ in range(self.num_iters):
            l_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.__sigmoid(l_model)

            dw = (1 / n_samples) * np.dot(X.T, (y-y_predicted))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #updating weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X, thresh = 0.5):
        l_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.__sigmoid(l_model)
        y_predicted_class = [1 if i > thresh else 0 for i in y_predicted]

        return y_predicted_class


    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))
