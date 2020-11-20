import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Normalizer import Normalizer

class DatasetHandler:
	def __init__(self, datasetName):
		self.datasetName = datasetName
		self.dataset = pd.read_csv(f"Data/{datasetName}.csv") # Import dataset
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.normalizer_train = Normalizer()
		self.normalizer_test = Normalizer()

	def getNormalizedData(self, train_size=0.75):
		# Split dataset for training and testing
		self.X_train, self.X_test, self.y_train, self.y_test = self.splitDataset(self.dataset, train_size)

		# Normalize data
		return self.normalizeData()

	
	def getTargetData(self):
		return np.array(self.y_test)


	def normalizeData(self):
		# Turn training data into np.arrays
		matrix_X_train = np.array(self.X_train)
		matrix_Y_train = np.array(self.y_train)

		# Normalize training data
		self.normalizer_train.fit(matrix_X_train, matrix_Y_train)
		normalizedX_train, normalizedY_train = self.normalizer_train.normalize(matrix_X_train, matrix_Y_train)

		# Turn test data into np.arrays
		matrix_X_test = np.array(self.X_test)
		matrix_Y_test = np.array(self.y_test)

		# Normalize testing data
		self.normalizer_test.fit(matrix_X_test, matrix_Y_test)
		normalizedX_test, normalizedY_test = self.normalizer_test.normalize(matrix_X_test, matrix_Y_test)

		return normalizedX_train, normalizedY_train, normalizedX_test, normalizedY_test


	def splitDataset(self, dataset, train_size):
		return train_test_split(dataset.iloc[ :,: -1], dataset.iloc[ :, -1:], random_state=0, train_size=train_size)