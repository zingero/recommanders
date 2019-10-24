import pickle
import numpy as np

from predictors.abstractpredictor import AbstractPredictor


class UniformPredictor(AbstractPredictor):
	def __init__(self):
		self.name = "Uniform"

	def predict(self, testing_set_file_path, possible_values):
		with open(testing_set_file_path, 'rb') as testing_set_file:
			testing_set = pickle.load(testing_set_file)
			rows_non_zero, cols_non_zero = testing_set.nonzero()
			mae = 0
			for i in range(len(rows_non_zero)):
				user, item = rows_non_zero[i], cols_non_zero[i]
				actual_rating = testing_set[user, item]
				predicted_rating = possible_values[np.random.randint(0, len(possible_values))]
				mae += abs(actual_rating - predicted_rating) / len(rows_non_zero)
			return mae
