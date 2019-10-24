import abc


class AbstractPredictor(object):
	def train(self, training_set_file_path):
		pass

	@abc.abstractmethod
	def predict(self, testing_set_file_path, possible_values):
		pass
