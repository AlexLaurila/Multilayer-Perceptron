import numpy as np

class Node():
	def __init__(self):
		self.output = None
		self.weights = []
		self.bias = np.random.random()


	def update_weights(self, dw, learning_rate):
		# Update weights
		self.weights -= dw * learning_rate
		

	def update_bias(self, db, learning_rate):
		# Update bias
		self.bias -=  db * learning_rate
