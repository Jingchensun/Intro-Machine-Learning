import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the LogisticRegression model.

        Parameters:
        - learning_rate (float): The step size at each iteration.
        - n_iterations (int): Number of iterations over the training dataset.

        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # To be initialized in fit method
        self.bias = None  # To be initialized in fit method

    def _sigmoid(self, z):
        """
        compute the sigmoid function.

        Parameters:
        - z (np.array): Linear combination of weights and features plus bias.

        Returns:
        - np.array: Sigmoid of z.
        """


    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X (np.array): Training features.
        - y (np.array): Target values.
        
        Returns:
        - self: The instance of the model.

        """
        # Initialize weights and bias to zeros
        
        # Gradient descent to update weights and bias
        
        # Implement the Gradient descent method to update the cost function
        

    def predict_proba(self, X):
        """
        Predict probability estimates for all classes.

        Parameters:
        - X (np.array): Test features.

        Returns:
        - np.array: Probability of the sample for each class in the model.
        """


    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.

        Parameters:
        - X (np.array): Test features.
        - threshold (float): Threshold used to convert probabilities into binary output.

        Returns:
        - np.array: Predicted class label per sample.
        """


# Load and preprocess dataset

# Split the data

# Feature scaling

# Training the model

# Calculate the metrics using scikit-learn

# Plotting the decision boundary and save the figure.




