import pickle
import numpy as np


class UserBasedPredictor(object):
	def __init__(self):
		self.user_distributions = dict()

	def predict(self, testing_set_file_path):
		with open(testing_set_file_path, 'rb') as testing_set_file:
			testing_set = pickle.load(testing_set_file)
			rows_non_zero, cols_non_zero = testing_set.nonzero()
			mae = 0
			for i in range(len(rows_non_zero)):
				user, item = rows_non_zero[i], cols_non_zero[i]
				actual_rating = testing_set[user, item]
				train_ratings_for_user = self.user_distributions[user]
				predicted_rating = train_ratings_for_user[np.random.randint(0, len(train_ratings_for_user))]
				mae += abs(actual_rating - predicted_rating) / len(rows_non_zero)
			return mae

	def train(self, training_set_file_path):
		with open(training_set_file_path, 'rb') as training_set_file:
			training_set = pickle.load(training_set_file)
			rows_non_zero, cols_non_zero = training_set.nonzero()
			for i in range(len(rows_non_zero)):
				current_user, current_item = rows_non_zero[i], cols_non_zero[i]
				if current_user not in self.user_distributions.keys():
					self.user_distributions[current_user] = []
				self.user_distributions[current_user].append(training_set[current_user, current_item])