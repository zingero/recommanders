import os

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import pickle

import tongue


class Parser(object):
	def parseDataToList(self, file_path):
		with open(file_path, "r") as f:
			data_file_list = f.readlines()
		for line in range(len(data_file_list)):
			data_file_list[line] = data_file_list[line].split(
				'\t')  # get rid of \t and neatly organize different columns in lists
			data_file_list[line][0] = int(data_file_list[line][0]) - 1  # make user ids start from 0
			data_file_list[line][1] = int(data_file_list[line][1]) - 1  # make item ids start from 0
			data_file_list[line][2] = float(data_file_list[line][2])  # ratings are incremented in steps of 0.5
			data_file_list[line][3] = int(data_file_list[line][3][:-1])  # get rid of the last \n
		return data_file_list  # data_file_list[j] is list representing line j. index 0 in list is user id, index 1 is item id, index 2 is rating, index 3 is time

	def getNumOfUsers(self, data_file_list):
		return max([data_file_list[j][0] for j in range(len(data_file_list))]) + 1  # +1 since they start from 0

	def getNumOfItems(self, data_file_list):
		return max([data_file_list[j][1] for j in range(len(data_file_list))]) + 1  # +1 since they start from 0

	def createRatingMatrix(self, data_file_list):
		num_of_users = self.getNumOfUsers(data_file_list)
		num_of_items = self.getNumOfItems(data_file_list)

		ratings_matrix = lil_matrix((num_of_users, num_of_items))
		for line in data_file_list:
			current_user_id = line[0]
			current_item_id = line[1]
			current_rating = line[2]
			ratings_matrix[current_user_id, current_item_id] = current_rating
		ratings_matrix = csr_matrix(ratings_matrix)
		return ratings_matrix

	def createTrainTestSplit(self, ratings_matrix, training_percentage=80):
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

		training_set = csr_matrix(training_set)
		testing_set = csr_matrix(testing_set)
		return training_set, testing_set


if __name__ == "__main__":
	parser = Parser()
	data_file_list = parser.parseDataToList(file_path='ml-100k/u.data')
	ratings_matrix = parser.createRatingMatrix(data_file_list)
	training_set, testing_set = parser.createTrainTestSplit(ratings_matrix, 80)

	with open(os.path.join(tongue.PARSED_DATA_PATH, 'ratings_matrix.p', 'wb')) as f:
		pickle.dump(ratings_matrix, f)
	with open(os.path.join(tongue.PARSED_DATA_PATH, 'training_set.p', 'wb')) as f:
		pickle.dump(training_set, f)
	with open(os.path.join(tongue.PARSED_DATA_PATH, 'testing_set.p', 'wb')) as f:
		pickle.dump(testing_set, f)
