import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

from Layer import Layer
import Functions

# MLP Class
class MLP:
	def __init__(self, loss_function):
		self.layers = []
		self.loss_function = loss_function
		self.lossList = []
	

	def get_loss(self):
		return np.sum(self.lossList) / len(self.lossList)	


	def get_layers(self):
		return len(self.layers)


	def add_layer(self, layerSize, activation_function):
		# Hidden & Output layer
		if (len(self.layers) >= 1):
			n_inputs = self.layers[-1].n_nodes
		# Only Input layer
		else:
			n_inputs = 0

		# Create the new layer
		newLayer = Layer(layerSize, n_inputs, activation_function)

		# Add the new layer to the list
		self.layers.append(newLayer)
	

	def _backprop(self, x, y, d_loss, learning_rate):
		# Loop backwards along the layers
		for i, layer in reversed(list(enumerate(self.layers))):
			# When in input layer
			if (i == 0):
				return
			# Output & Hidden layers
			else:
				input = self.layers[i-1].output
				d_loss = layer.backprop(d_loss, learning_rate, y, input)


	def train(self, dataset, x, y, learning_rate=0.1, n_epochs=100, decay=0):
		for i in range(n_epochs):
			if (i % 10 == 0):
				print(f"{dataset} epoch: {i}")
			if(x.ndim == 2):
				x, y = shuffle(x, y)
				for i, row in enumerate(x):
					# Make prediction
					pred = self.predict(row, y[i])
					# Calculate loss
					loss = self.loss_function.forward(pred, y[i])
					# Get derivative of loss
					d_loss = self.loss_function.backward(pred, y[i])
					# Backpropagate
					self._backprop(row, y[i], d_loss, learning_rate)
					# Learning rate decay
					learning_rate = learning_rate * (1/1 + decay)
					learning_rate = max(learning_rate, 0.01)
			else:
				# Make prediction
				pred = self.predict(x, y)
				# Calculate loss
				loss = self.loss_function.forward(pred, y)
				# Get derivative of loss
				d_loss = self.loss_function.backward(pred, y)
				# Backpropagate
				self._backprop(x, y, d_loss, learning_rate)


	def predict(self, x, y):
		predictions = []

		# Multiple inputs iterated
		if(x.ndim == 2):
			for indexY, row in enumerate(x):
				for i, layer in enumerate(self.layers):
					if (i == 0):
						layer.forward(row)
					else:
						layer.forward(self.layers[i-1].output)
				# Store prediction
				predictions.append(self.layers[-1].output[0])
				# Store loss
				self.lossList.append(self.loss_function.forward(self.layers[-1].output[0], y[indexY]))
		# One input only
		else:
			for i, layer in enumerate(self.layers):
				if (i == 0):
					layer.forward(x)
				else:
					layer.forward(self.layers[i-1].output)
			# Store prediction
			predictions.append(self.layers[-1].output[0])
			# Store loss
			self.lossList.append(self.loss_function.forward(self.layers[-1].output[0], y))

		return predictions 