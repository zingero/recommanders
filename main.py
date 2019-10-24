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
import userbasedpredictor
import moviebasedpredictor
import combinedpredictor
import matrixfactorizationpredictor


class Main(object):
	def __init__(self):
		self.init_logging()
		recommender_environment = {tongue.ML_100K: {'parser': Parser100K, 'data_file_name': tongue.ML_100K_FILE_NAME, 'possible_values': numpy.arange(1, 6)},
									tongue.ML_20M: {'parser': Parser20M, 'data_file_name': tongue.ML_20M_FILE_NAME, 'possible_values': numpy.arange(1, 5.5, 0.5)}}
		for data_set_folder_name, environment in recommender_environment.items():
			if not os.path.exists(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name)):
				os.makedirs(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name))
				self.init_parser(data_set_folder_name, environment)
			self.init_uniform_predictor(data_set_folder_name, environment)
			self.init_user_based_predictor(data_set_folder_name, environment)
			self.init_movie_based_predictor(data_set_folder_name, environment)
			self.init_combined_predictor(data_set_folder_name, environment)
			self.init_matrix_factorization_predictor(data_set_folder_name)

	def init_logging(self):
		logging.basicConfig(format = '%(asctime)s %(levelname)s %(message)s', level = logging.DEBUG)

	def init_parser(self, data_set_folder_name, environment):
		os.makedirs(tongue.PARSED_DATA_PATH, exist_ok=True)
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
		predictor = uniformpredictor.UniformPredictor()
		self.init_generic_predictor(data_set_folder_name, predictor, environment)

	def init_user_based_predictor(self, data_set_folder_name, environment):
		predictor = userbasedpredictor.UserBasedPredictor()
		self.init_generic_predictor(data_set_folder_name, predictor, environment)

	def init_movie_based_predictor(self, data_set_folder_name, environment):
		predictor = moviebasedpredictor.MovieBasedPredictor()
		self.init_generic_predictor(data_set_folder_name, predictor, environment)

	def init_combined_predictor(self, data_set_folder_name, environment):
		predictor = combinedpredictor.CombinedPredictor()
		self.init_generic_predictor(data_set_folder_name, predictor, environment)

	def init_matrix_factorization_predictor(self, data_set_folder_name):
		predictor = matrixfactorizationpredictor.MatrixFactorizationPredictor()
		self.init_generic_predictor(data_set_folder_name, predictor)

	def init_generic_predictor(self, data_set_folder_name, predictor, environment = {}):
		logging.info("(%s) Predicting: %s" % (predictor.name, data_set_folder_name))
		start_time = time.time()
		predictor.train(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.TRAINING_SET_FILE_NAME))
		logging.info("(%s) Finished training for: %s. trained in: %s seconds" % (predictor.name, data_set_folder_name, time.time() - start_time))
		start_time = time.time()
		mae = predictor.predict(os.path.join(tongue.PARSED_DATA_PATH, data_set_folder_name, tongue.TESTING_SET_FILE_NAME), environment.get('possible_values'))
		logging.info("(%s) Finished prediction for: %s. mae: %s. predicted in: %s seconds" % (predictor.name, data_set_folder_name, mae, time.time() - start_time))

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
