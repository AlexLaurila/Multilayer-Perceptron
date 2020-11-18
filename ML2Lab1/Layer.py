import pandas as pd
import numpy as np
import random

# Layer Class
class Layer:
	def __init__(self, n_nodes, n_inputs, activation_function):
		self.output = []
		self.biases = np.random.random(n_nodes)
		self.weights = np.random.randn(n_inputs, n_nodes)
		self.activation_function = activation_function
		self.n_nodes = n_nodes
		self.n_inputs = n_inputs


	def forward(self, x):
		if (self.n_inputs > 0):
			z = np.dot(x, self.weights) + self.biases
		else:
			z = x

		self.output = self.activation_function.forward(z)

		
	def backprop(self, d_loss, learning_rate, y):
		d_sig = self.activation_function.backward(self.output)
		dw = np.sum(self.output * d_loss * d_sig / len(y))
		db = np.sum(d_loss * d_sig / len(y))

		d_loss = self.weights * d_loss * d_sig

		# Update own weights & biases
		self.weights -= dw * learning_rate
		self.biases -= db * learning_rate
