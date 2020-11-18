import numpy as np

class Normalizer:
	def __init__(self):
		self.mean = []
		self.std = []

	def fit(self, x, y):
		for i in range(x.shape[1]):
			self.mean.append(np.mean(x[:, i]))
			self.std.append(np.std(x[:, i]))

		self.mean.append(np.mean(y))
		self.std.append(np.std(y))

		self.mean = np.array(self.mean)
		self.std = np.array(self.std)

	def normalize(self, x, y):
		# Standardizes all values to mean 0, std 1.
		return (x - self.mean[:-1]) / self.std[:-1], (y - self.mean[-1]) / self.std[-1]

	def renormalize(self, y):
		# Scales y-values (predictions) back to original scale.
		return y * self.std[-1] + self.mean[-1]
