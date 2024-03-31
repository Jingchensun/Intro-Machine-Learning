import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        """
        Fit the SVM model to training data.
        
        Parameters:
        - X (np.array): Training features.
        - y (np.array): Target values.

        Returns:
        - None
        """
        # Initialize weights and bias to zeros
        # Implement gradient descent to update weights and bias


    def predict(self, X):

        """
        Predict class labels for samples in X.

        Parameters:
        - X (np.array): Test features.

        Returns:
        - np.array: Predicted class label per sample.
        """
        # Compute the linear combination of weights and features plus bias
        # Return class labels based on the sign of the linear combination


# Load and preprocess dataset

# Split the data

# Feature scaling

# Training the model

# Calculate the metrics using scikit-learn

# Plotting the decision boundary and save the figure.