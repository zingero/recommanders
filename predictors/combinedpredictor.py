from predictors.moviebasedpredictor import MovieBasedPredictor
from predictors.userbasedpredictor import UserBasedPredictor
from predictors.userormoviebasedpredictor import UserOrMovieBasedPredictor


class CombinedPredictor(UserOrMovieBasedPredictor):
	def __init__(self):
		super().__init__()
		self.user_based_predictor = UserBasedPredictor()
		self.item_based_predictor = MovieBasedPredictor()

	def name(self):
		return "Combined"

	def add_training_set_distribution(self, user, item, training_set):
		self.user_based_predictor.add_training_set_distribution(user, item, training_set)
		self.item_based_predictor.add_training_set_distribution(user, item, training_set)

	def calculate_predicted_rating(self, user, item, possible_values):
		user_predicted_rating = self.user_based_predictor.calculate_predicted_rating(user, item, possible_values)
		item_predicted_rating = self.item_based_predictor.calculate_predicted_rating(user, item, possible_values)
		return (user_predicted_rating + item_predicted_rating) / 2

