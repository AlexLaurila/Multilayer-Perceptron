import numpy as np

class SquaredErrorLossFunction:

	@staticmethod
	def forward(predictions, correct_outputs):
		return (predictions - correct_outputs) ** 2

	@staticmethod
	def backward(predictions, correct_outputs):
		return 2 * (predictions - correct_outputs)


class LinearActivationFunction:
	
	@staticmethod
	def forward(z):
		return z
	
	@staticmethod
	def backward(z):
		return np.ones(z.size)


class InputActivationFunction:

	@staticmethod
	def forward(z): 
		return z

	@staticmethod
	def backward(z): 
		return np.zeros(z.size)


class SigmoidActivationFunction:
	
	@staticmethod
	def forward(z):
		return 1 / (1 + np.exp(-z))
	
	@staticmethod
	def backward(z):
		return z * (1 - z)