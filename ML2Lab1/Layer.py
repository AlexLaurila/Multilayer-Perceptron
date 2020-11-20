import pandas as pd
import numpy as np
import random

from Node import Node

# Layer Class
class Layer:
	def __init__(self, n_nodes, n_inputs, activation_function):
		self.nodes = []
		self.activation_function = activation_function
		self.n_nodes = n_nodes
		self.n_inputs = n_inputs

		for i in range(self.n_nodes):
			self.nodes.append(Node())


	def forward(self, x):
		# Hidden & Output layer
		if (self.n_inputs > 0):
			pLNodes = x

			# Iterate over nodes in current layer
			for i, node in enumerate(self.nodes):
				z = 0
				# Iterate over nodes in previous layer to get values and weights
				for pLNode in pLNodes:
					z += pLNode.output * pLNode.weights[i]
				# Evaluate finished output value
				node.output = self.activation_function.forward(z + node.bias)
		# Only Input layer
		else:
			# Iterate over inputvalues and nodes in input layer
			for input, node in zip(x, self.nodes):
				# Feed one input into input layer-node output
				node.output = self.activation_function.forward(input)

		
	def backprop(self, d_loss, learning_rate, y, p_Layer):
		currentLayer_output = []
		# Iterate current layer nodes
		for node in self.nodes:
			currentLayer_output.append(node.output)
		
		# Get d_sig from current layer output
		d_sig = self.activation_function.backward(np.array(currentLayer_output)) # för varje nod uppdateras

		previousLayer_output = []
		previousLayer_weights = []
		for node in p_Layer.nodes:
			previousLayer_output.append(node.output)
			previousLayer_weights.append(node.weights)

		dw = []
		for i, output in enumerate(previousLayer_output):
			dw_list = []
			for j in range(len(self.nodes)):
				dw_list.append(output * d_loss[j] * d_sig[j])
			dw.append(dw_list)
		
		db = []
		for i in range(self.n_nodes):
			db.append(np.sum(d_loss[i] * d_sig[i]))

		# Set new d_loss
		d_loss_list = []
		for i in range(len(p_Layer.nodes)):
			d_loss_list.append(np.sum(previousLayer_weights[i] * dw[i])) # d_loss is calculated with previous layer weights
		
		# Iterate over nodes in current layer (L)
		for i in range(self.n_nodes):
			# Iterate over nodes in previous layer (L-1)
			for j, node in enumerate(p_Layer.nodes):
				# Update weights and bias in node
				node.update_weights(dw[j][i], learning_rate)
			node.update_bias(db[i], learning_rate)

		return d_loss_list
