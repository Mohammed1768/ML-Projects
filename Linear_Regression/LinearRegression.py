import numpy as np
import pandas as pd

class LinearRegression:
    """
    Simple Linear Regression using Stochastic Gradient Descent (SGD).
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        """
        Train the model using SGD.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features (n_samples x n_features).
        Y : pandas.Series
            Training targets (n_samples,).
        """
        n, m = np.shape(X)
        th = np.random.rand(m)
        th0 = np.random.randint(1, 100)
        alpha = 0.01
        epochs = 1000

        for _ in range(epochs):
            for x, y in zip(X.values, Y.values):
                prediction = np.dot(x, th) + th0
                grad_th, grad_th0 = (prediction - y) * x, (prediction - y)
                th -= alpha * grad_th
                th0 -= alpha * grad_th0

        self.th, self.th0 = th, th0

    def predict(self, x):
        """
        Predict target values.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Input features.

        Returns
        -------
        numpy.ndarray
            Predictions.
        """
        return np.dot(x, self.th) + self.th0

    def test(self, X, Y):
        """
        Compute Mean Absolute Error (MAE).
        """
        return np.mean(np.abs(Y - self.predict(X)))