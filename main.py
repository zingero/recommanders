import os
import pickle

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

import parser
import tongue
import uniformpredictor


class Main(object):
	def __init__(self):
		os.makedirs(tongue.PARSED_DATA_PATH, exist_ok = True)
		self.init_parser()
		self.init_uniform_predictor()

	def init_parser(self):
		self.__parser = parser.Parser(file_path = 'ml-100k/u.data')
		ratings_matrix = self.__parser.create_ratings_matrix()
		training_set, testing_set = self.__split_ratings_matrix_to_training_and_testing(ratings_matrix, 80)
		with open(os.path.join(tongue.PARSED_DATA_PATH, tongue.RATING_MATRIX_FILE_NAME), 'wb') as f:
			pickle.dump(ratings_matrix, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, tongue.TRAINING_SET_FILE_NAME), 'wb') as f:
			pickle.dump(training_set, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, tongue.TESTING_SET_FILE_NAME), 'wb') as f:
			pickle.dump(testing_set, f)

	def init_uniform_predictor(self):
		self.uniform_predictor = uniformpredictor.UniformPredictor()
		mae = self.uniform_predictor.predict(tongue.TESTING_SET_FILE_NAME)
		print("mae: %s" % mae)

	def __split_ratings_matrix_to_training_and_testing(self, ratings_matrix, training_percentage = 80):
		training_set, testing_set = lil_matrix(ratings_matrix.shape), lil_matrix(ratings_matrix.shape)
		rows_non_zero, cols_non_zero = ratings_matrix.nonzero()
		for i in range(len(rows_non_zero)):
			current_row, current_col = rows_non_zero[i], cols_non_zero[i]
			current_rating = ratings_matrix[current_row, current_col]
			random_roll = numpy.random.randint(0, 100)
			if random_roll <= training_percentage:
				training_set[current_row, current_col] = current_rating
			else:
				testing_set[current_row, current_col] = current_rating

		return csr_matrix(training_set), csr_matrix(testing_set)


if __name__ == "__main__":
	Main()
