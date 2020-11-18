import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from MLP import MLP
from Layer import Layer
from Normalizer import Normalizer
import Functions

# Dataset to use
Dataset = "laser" 

# Import dataset
dataset = pd.read_csv(f"Data/{Dataset}.csv")

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[ :,: -1], dataset.iloc[ :, -1:], random_state=0, train_size=0.01)

# Turn training data into np.arrays
matrix_X = np.array(X_train)
matrix_Y = np.array(y_train)

# Normalize the data
normalizer = Normalizer()
normalizer.fit(matrix_X, matrix_Y)
normalizedX, normalizedY = normalizer.normalize(matrix_X, matrix_Y)

# Prepare a new Neural Network
neuralNetwork = MLP(Functions.SquaredErrorLossFunction())

# Add layers to the Neural Network
neuralNetwork.add_layer(X_train.shape[1], Functions.InputActivationFunction()) #Input
neuralNetwork.add_layer(16, Functions.SigmoidActivationFunction()) #Hidden layer
neuralNetwork.add_layer(16, Functions.SigmoidActivationFunction()) #Hidden layer
neuralNetwork.add_layer(16, Functions.SigmoidActivationFunction()) #Hidden layer
neuralNetwork.add_layer(1, Functions.LinearActivationFunction()) #Output layer

# Train the Neural Network
neuralNetwork.train(normalizedX[0], normalizedY[0])

# Predict
predicted = neuralNetwork.predict(normalizedX[0], normalizedY[0]) #Få tillbaka en lista av Y?
predicted = np.array(predicted)
predictedRenomalized = normalizer.renormalize(predicted)
print(f"Predicted: {predictedRenomalized}")
print(f"Target: {matrix_Y[0]}")