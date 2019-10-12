import os
import pickle
import time
import logging

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

from parsers.parser100k import Parser100K
from parsers.parser20m import Parser20M
import tongue
import uniformpredictor


class Main(object):
	def __init__(self):
		self.init_logging()
		os.makedirs(tongue.PARSED_DATA_PATH, exist_ok = True)
		recommender_environment = {tongue.ML_100K: {'parser': Parser100K, 'data_file_name': tongue.ML_100K_FILE_NAME, 'possible_values': range(1, 6)},
									tongue.ML_20M: {'parser': Parser20M, 'data_file_name': tongue.ML_20M_FILE_NAME, 'possible_values': range(1, 11)}}
		for data_set_folder_name, environment in recommender_environment.items():
			if not os.path.exists(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name)):
				os.makedirs(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name))
				self.init_parser(data_set_folder_name, environment)
			self.init_uniform_predictor(data_set_folder_name, environment)

	def init_logging(self):
		logging.basicConfig(format = '%(asctime)s %(levelname)s %(message)s', level = logging.DEBUG)

	def init_parser(self, data_set_folder_name, environment):
		start_time = time.time()
		logging.info("Parsing: %s" % data_set_folder_name)
		parser = environment['parser'](file_path = os.path.join(data_set_folder_name, environment['data_file_name']))
		ratings_matrix = parser.create_ratings_matrix()
		training_set, testing_set = self.__split_ratings_matrix_to_training_and_testing(ratings_matrix, 80)
		with open(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.RATING_MATRIX_FILE_NAME), 'wb') as f:
			pickle.dump(ratings_matrix, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.TRAINING_SET_FILE_NAME), 'wb') as f:
			pickle.dump(training_set, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.TESTING_SET_FILE_NAME), 'wb') as f:
			pickle.dump(testing_set, f)
		logging.info("Parsed in: %s seconds" % (time.time() - start_time))

	def init_uniform_predictor(self, data_set_folder_name, environment):
		start_time = time.time()
		logging.info("(Uniform) Predicting: %s" % data_set_folder_name)
		uniform_predictor = uniformpredictor.UniformPredictor()
		mae = uniform_predictor.predict(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.TESTING_SET_FILE_NAME), environment['possible_values'])
		logging.info("(Uniform) Finished prediction for: %s. mae: %s. predicted in: %s seconds" % (data_set_folder_name, mae, time.time() - start_time))

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
