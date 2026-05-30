import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit (self, X, y):
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            linear = X @ self.W + self.b
            y_pred = self.sigmoid(linear)

            dW = (1/n_samples) * X.T @ (y_pred-y);
            db = (1/n_samples) * np.sum(y_pred-y);

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self, X):
        linear = X @ self.W + self.b
        return self.sigmoid(linear)

    def predict(self, X):
        return ((self.predict_proba(X)>=0.5).astype(int))

