import numpy as np

from predictors.userormoviebasedpredictor import UserOrMovieBasedPredictor


class UserBasedPredictor(UserOrMovieBasedPredictor):
	def name(self):
		return "User based"

	def add_training_set_distribution(self, user, item, training_set):
		try:
			self.distributions[user].append(training_set[user, item])
		except KeyError:
			self.distributions[user] = [training_set[user, item]]

	def calculate_predicted_rating(self, user, item, possible_values):
		try:
			train_ratings_for_user = self.distributions[user]
			return train_ratings_for_user[np.random.randint(0, len(train_ratings_for_user))]
		except:
			return possible_values[np.random.randint(0, len(possible_values))]
