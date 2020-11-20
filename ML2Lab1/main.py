import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from itertools import chain

from MLP import MLP
from Layer import Layer
from Normalizer import Normalizer
from DatasetHandler import DatasetHandler
from Plotter import Plotter
import Functions

"""
Available Datasets: boston, concrete, friedm, istanbul, laser, plastic, quakes, stock, wizmir
"""

# Selected dataset
dataset = "boston"

# Get Normalized Test & Train Data
datasetHandler = DatasetHandler(dataset)
normalizedX_train, normalizedY_train, normalizedX_test, normalizedY_test = datasetHandler.getNormalizedData()

# Create Neural Network
neuralNetwork = MLP(Functions.SquaredErrorLossFunction())

# Add layers to the Neural Network
neuralNetwork.add_layer(datasetHandler.X_train.shape[1], Functions.InputActivationFunction()) # Input layer
neuralNetwork.add_layer(6, Functions.SigmoidActivationFunction()) # Hidden layer
neuralNetwork.add_layer(8, Functions.SigmoidActivationFunction()) # Hidden layer
neuralNetwork.add_layer(4, Functions.SigmoidActivationFunction()) # Hidden layer
neuralNetwork.add_layer(1, Functions.LinearActivationFunction()) # Output layer

# Train the Neural Network
neuralNetwork.train(normalizedX_train, normalizedY_train, 0.1, 100, True, -0.1)

# Predict with testing data in the Neural Network
predicted = neuralNetwork.predict(normalizedX_test, normalizedY_test)

# Renormalize results
predicted = np.array(predicted)
predictedRenomalized = datasetHandler.normalizer_test.renormalize(predicted)

# Fetch TargetData
target = datasetHandler.getTargetData()

# Plot training data
max = max(chain(predictedRenomalized, target))
# Set plot axis limits
plt.xlim(0, max+1)
plt.ylim(0, max+1)
# Plot
plt.scatter(predictedRenomalized, target)
plt.show()
