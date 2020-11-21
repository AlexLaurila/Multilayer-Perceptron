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
		# Hiden & Output layer
		if (self.n_inputs > 0):
			z = np.dot(x, self.weights) + self.biases
		# Only Inpyt layer
		else:
			z = x

		# Get finished output with Activation Function
		self.output = self.activation_function.forward(z)


	def backprop(self, d_loss, learning_rate, y, input):
		# Get d_sig from current layer output
		d_sig = self.activation_function.backward(self.output)
		
		dw = []
		for i in range(self.n_inputs):
			dw.append(np.dot(input[i], d_loss) * d_sig)

		db = d_loss * d_sig

		d_loss_prev = []
		for i in range(self.n_inputs): 
			d_loss_prev.append(np.sum(self.weights[i] * dw[i])) 

		self.weights -= np.dot(dw, learning_rate)
		self.biases -= db * learning_rate

		return d_loss_prev   
