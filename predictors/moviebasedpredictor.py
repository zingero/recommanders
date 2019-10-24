import numpy as np

from predictors.userormoviebasedpredictor import UserOrMovieBasedPredictor


class MovieBasedPredictor(UserOrMovieBasedPredictor):
	def name(self):
		return "Movie based"

	def add_training_set_distribution(self, user, item, training_set):
		try:
			self.distributions[item].append(training_set[user, item])
		except KeyError:
			self.distributions[item] = [training_set[user, item]]

	def calculate_predicted_rating(self, user, item, possible_values):
		try:
			train_ratings_for_item = self.distributions[item]
			return train_ratings_for_item[np.random.randint(0, len(train_ratings_for_item))]
		except:
			return possible_values[np.random.randint(0, len(possible_values))]
