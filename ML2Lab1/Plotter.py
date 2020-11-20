from matplotlib import pyplot as plt

class Plotter:
	def __init__(self):
		pass

	def Plot(self, X, y):
		for i in range(X.shape[0]):
			plt.scatter(X[i], y[i])
		plt.show()
