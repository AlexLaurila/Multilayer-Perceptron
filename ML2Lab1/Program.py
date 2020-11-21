import numpy as np
import time

from MLP import MLP
from DatasetHandler import DatasetHandler
from Info import Info
import Functions

class Program:
	def Run(self, dataset, i):
		# Input Parameters
		n_epochs = 100
		train_size = 0.75
		learning_rate = 0.1
		decay = -0.2

		# Get Normalized Test & Train Data
		datasetHandler = DatasetHandler(dataset)
		normalizedX_train, normalizedY_train, normalizedX_test, normalizedY_test = datasetHandler.getNormalizedData(train_size)

		# Create Neural Network
		neuralNetwork = MLP(Functions.SquaredErrorLossFunction())

		# Add layers to the Neural Network
		neuralNetwork.add_layer(datasetHandler.X_train.shape[1], Functions.InputActivationFunction()) # Input layer
		neuralNetwork.add_layer(8, Functions.SigmoidActivationFunction()) # Hidden layer
		neuralNetwork.add_layer(8, Functions.SigmoidActivationFunction()) # Hidden layer
		neuralNetwork.add_layer(8, Functions.SigmoidActivationFunction()) # Hidden layer
		neuralNetwork.add_layer(1, Functions.LinearActivationFunction()) # Output layer

		tStartTrain = time.time()
		# Train the Neural Network
		neuralNetwork.train(dataset, normalizedX_train, normalizedY_train, learning_rate, n_epochs, decay)
		tEndTrain = time.time()
		trainTime = tEndTrain - tStartTrain

		tStartTest = time.time()
		# Predict with testing data in the Neural Network
		predicted = neuralNetwork.predict(normalizedX_test, normalizedY_test)
		tEndTest = time.time()
		testTime = tEndTest - tStartTest

		print(f"{dataset} training finished!")

		# Renormalize results
		predicted = np.array(predicted)
		predictedRenomalized = datasetHandler.normalizer_test.renormalize(predicted)

		# Fetch TargetData
		target = datasetHandler.getTargetData()

		# Calculated loss
		loss = neuralNetwork.get_loss();
		n_layers = neuralNetwork.get_layers();

		# Information
		InfoList = Info(predictedRenomalized, target, loss, dataset, trainTime, testTime, n_epochs, n_layers, learning_rate, decay)
		
		return InfoList