import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#################
    # ID3 Algorithm
#################

def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    """
    pass  

def InfoGain(data, split_attribute_name, target_name="class"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default value is "class"
    """
    pass  

def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class = None):
    """
    ID3 Algorithm: This function takes five parameters:
    1. data = the dataset for which the ID3 algorithm should be run — in the first run, this should be the entire dataset
    2. originaldata = this is the original dataset needed to calculate the mode target feature value of the original dataset
    in the first run
    3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset — slicing the feature space
    4. target_attribute_name = the name of the target attribute
    5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is
    also needed for the recursive call in case the dataset delivered to the ID3 algorithm consists only of samples with the same class. Then,
    this function returns the mode target feature value of the parent node dataset as the class value of the dataset.
    """
    pass  

# Load and prepare the dataset

# Split the dataset for training and testing

# Prepare features

# Train ID3 model

# Predict and evaluate the model 



#################
    # C 4.5 Algorithm
#################

def entropy(target_col):
    """
    Calculate the entropy of a dataset for a given target column.
    """
    pass  

def info_gain(data, split_attribute_name, target_name="class"):
    """
    Calculate the information gain of splitting the dataset on a specific attribute.
    """
    pass  

def gain_ratio(data, split_attribute_name, target_name="class"):
    """
    Calculate the gain ratio for a dataset based on a given attribute, adjusting information gain for split info.
    """
    pass  

def best_split(data, features, target_name="class"):
    """
    Determine the best feature to split on in the dataset, based on the highest gain ratio.
    """
    pass  

def C45(data, features, target_name="class", parent_node_class=None):
    """
    Recursively build the decision tree using the C4.5 algorithm.
    """
    pass  

# Load and preprocess the Iris dataset
def load_and_prepare_data():
    """
    Load the Iris dataset, preprocess it, and split it into training and testing sets.
    """
    pass  

# Train the C4.5 model on the Iris dataset
def train_model(X_train, y_train, features):
    """
    Train the C4.5 decision tree model on the training dataset.
    """
    pass 

# Predict the class of samples using the trained C4.5 model
def predict(tree, instance):
    """
    Predict the class label for a single instance using the trained C4.5 decision tree.
    """
    pass  

# Evaluate the performance of the C4.5 model on the test dataset
def evaluate_model(tree, X_test, y_test):
    """
    Evaluate the metrics of the C4.5 model on the test dataset.
    """
    pass  


# Load and prepare the dataset

# Define features

# Train the C4.5 model

# Predict and evaluate the model 
