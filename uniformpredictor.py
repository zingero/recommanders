import pickle
import numpy as np


class UniformPredictor(object):
	def predict(self, testing_set_file_path, possible_values):
		with open(testing_set_file_path, 'rb') as testing_set_file:
			testing_set = pickle.load(testing_set_file)
			rows_non_zero, cols_non_zero = testing_set.nonzero()
			mae = 0
			for i in range(len(rows_non_zero)):
				current_rating = testing_set[rows_non_zero[i], cols_non_zero[i]]
				current_prediction = possible_values[np.random.randint(0, len(possible_values))]
				mae += abs(current_rating - current_prediction) / len(rows_non_zero)
			return mae
