
class Info:
	def __init__(self, predictedRenomalized, target, loss, Dataset, trainTime, testTime, n_epochs, n_layers, learning_rate, decay):
		self.predictedRenomalized = predictedRenomalized
		self.target = target
		self.Dataset = Dataset
		self.loss = loss
		self.trainTime = trainTime
		self.testTime = testTime
		self.n_epochs = n_epochs
		self.n_layers = n_layers
		self.learning_rate = learning_rate
		self.decay = decay
