import os
import pickle

import parser
import tongue
import uniformpredictor


class Main(object):
	def __init__(self):
		print('hi')
		self.init_parser()
		self.init_uniform_predictor()

	def init_parser(self):
		self.parser = parser.Parser()
		data_file_list = self.parser.parseDataToList(file_path = 'ml-100k/u.data')
		ratings_matrix = self.parser.createRatingMatrix(data_file_list)
		training_set, testing_set = self.parser.createTrainTestSplit(ratings_matrix, 80)
		with open(os.path.join(tongue.PARSED_DATA_PATH, 'ratings_matrix.p'), 'wb') as f:
			pickle.dump(ratings_matrix, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, 'training_set.p'), 'wb') as f:
			pickle.dump(training_set, f)
		with open(os.path.join(tongue.PARSED_DATA_PATH, 'testing_set.p'), 'wb') as f:
			pickle.dump(testing_set, f)

	def init_uniform_predictor(self):
		self.uniform_predictor = uniformpredictor.UniformPredictor()
		mae = self.uniform_predictor.predict(tongue.TESTING_SET_FILE_PATH)
		print("mae: %s" % mae)


if __name__ == "__main__":
	Main()
