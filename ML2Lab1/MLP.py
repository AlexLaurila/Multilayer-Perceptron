import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

from Layer import Layer
from Normalizer import Normalizer
import Functions

# MLP Class
class MLP:
	def __init__(self, loss_function):
		self.layers = []
		self.loss_function = loss_function
		self.normalizer = Normalizer()
	

	def add_layer(self, n_nodes, activation_function):
		# Hidden & Output layer
		if (len(self.layers) >= 1):
			# The number of inputs for every node in the new layer = the number of nodes in previous layer
			n_inputs = self.layers[-1].n_nodes
		# Only Input layer
		else:
			# Input layer incoming connections
			n_inputs = 0

		# Create the new layer
		newLayer = Layer(n_nodes, n_inputs, activation_function)
		# Iterate over previous layer nodes to add correct amount of weights
		if (len(self.layers) >= 1):
			for node in self.layers[-1].nodes:
				node.weights = np.random.randn(n_nodes)

		# Add the new layer to the list
		self.layers.append(newLayer)
	

	def _backprop(self, x, y, d_loss, learning_rate):
		# Loop backwards along the layers
		for i, layer in reversed(list(enumerate(self.layers))):
			if (i == 0):
				return
			d_loss = layer.backprop(d_loss, learning_rate, y, self.layers[i-1])


	def train(self, x, y, learning_rate=0.01, n_epochs=100, learning_rate_decay=False, decay=0):
		for i in range(n_epochs):
			if (i % 10 == 0):
				print(f"Epoch: {i}")
			if(x.ndim == 2):
				x, y = shuffle(x, y, random_state=0)
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
					if (learning_rate_decay):
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

		if(x.ndim == 2):
			for row in x:
				for i, layer in enumerate(self.layers):
					if (i == 0):
						layer.forward(row)
					else:
						layer.forward(self.layers[i-1].nodes)
				predictions.append(self.layers[-1].nodes[-1].output)
		else:
			for i, layer in enumerate(self.layers):
				if (i == 0):
					layer.forward(x)
				else:
					layer.forward(self.layers[i-1].nodes)
			predictions.append(self.layers[-1].nodes[-1].output)

		return predictions 