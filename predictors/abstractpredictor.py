import abc


class AbstractPredictor(object):
	__meta__ = abc.ABCMeta

	@abc.abstractmethod
	def name(self):
		pass

	def train(self, training_set_file_path):
		pass

	@abc.abstractmethod
	def predict(self, testing_set_file_path, possible_values):
		pass
