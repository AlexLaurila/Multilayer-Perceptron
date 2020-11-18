import pandas as pd
import numpy as np
import random

from Layer import Layer
from Normalizer import Normalizer
import Functions

# MLP Class
class MLP:
	def __init__(self, loss_function):
		self.layers = []
		self.loss_function = loss_function
	

	def add_layer(self, layerSize, activation_function):
		if (len(self.layers) >= 1):
			n_inputs = self.layers[-1].n_nodes
		else:
			n_inputs = 0

		newLayer = Layer(layerSize, n_inputs, activation_function)
		self.layers.append(newLayer)
	

	def _backprop(self, x, y, d_loss, learning_rate):
		# Loop backwards along the layers
		for i, layer in reversed(list(enumerate(self.layers))):
			layer.backprop(d_loss, learning_rate, y)


	def train(self, x, y, learning_rate=0.01, n_epochs=100):
		for i in range(n_epochs):
			if (i % 10 == 0):
				print(f"Epoch: {i}")
			if(x.ndim == 2):
				for index, row in enumerate(x):
					# Make prediction
					pred = self.predict(row, y[index])
					# Calculate loss
					L = self.loss_function.forward(pred, y[index])
					print(f"Loss: {L}")
					# Get derivative of loss
					d_loss = self.loss_function.backward(pred, y[index])
					# Backpropogate
					self._backprop(row, y[index], d_loss, learning_rate)
			else:
				# Make prediction
				pred = self.predict(x, y)
				# Calculate loss
				L = self.loss_function.forward(pred, y)
				print(f"Loss: {L}")
				# Get derivative of loss
				d_loss = self.loss_function.backward(pred, y)
				# Backpropogate
				self._backprop(x, y, d_loss, learning_rate)


	def predict(self, x, y): 
		predictions = []

		for index, layer in enumerate(self.layers):
			if (index == 0):
				layer.forward(x)
			else:
				layer.forward(self.layers[index-1].output)

		predictions.append(self.layers[-1].output[0])

		return predictions