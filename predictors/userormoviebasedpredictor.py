import abc
import pickle

from predictors.abstractpredictor import AbstractPredictor


class UserOrMovieBasedPredictor(AbstractPredictor):
	__meta__ = abc.ABCMeta

	def __init__(self):
		self.distributions = dict()

	def train(self, training_set_file_path):
		with open(training_set_file_path, 'rb') as training_set_file:
			training_set = pickle.load(training_set_file)
			rows_non_zero, cols_non_zero = training_set.nonzero()
			for i in range(len(rows_non_zero)):
				user, item = rows_non_zero[i], cols_non_zero[i]
				self.add_training_set_distribution(user, item, training_set)

	def predict(self, testing_set_file_path, possible_values):
		with open(testing_set_file_path, 'rb') as testing_set_file:
			testing_set = pickle.load(testing_set_file)
			rows_non_zero, cols_non_zero = testing_set.nonzero()
			mae = 0
			for i in range(len(rows_non_zero)):
				user, item = rows_non_zero[i], cols_non_zero[i]
				actual_rating = testing_set[user, item]
				predicted_rating = self.calculate_predicted_rating(user, item, possible_values)
				mae += abs(actual_rating - predicted_rating) / len(rows_non_zero)
			return mae

	@abc.abstractmethod
	def add_training_set_distribution(self, user, item, training_set):
		pass

	@abc.abstractmethod
	def calculate_predicted_rating(self, user, item, possible_values):
		pass
