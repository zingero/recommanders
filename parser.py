from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

import tongue
from data import Data


class Parser(object):
	def __init__(self, file_path):
		self.__data = self.__parse_file(file_path)

	def __parse_file(self, file_path):
		with open(file_path, "r") as f:
			data_file_list = f.readlines()
		data = Data()
		for line in data_file_list:
			splitted_line = line.strip().split('\t')  # get rid of \t and neatly organize different columns in lists
			splitted_line[tongue.USER_ID_INDEX] = int(splitted_line[tongue.USER_ID_INDEX]) - 1  # make user ids start from 0
			splitted_line[tongue.ITEM_ID_INDEX] = int(splitted_line[tongue.ITEM_ID_INDEX]) - 1  # make item ids start from 0
			splitted_line[tongue.RATING_INDEX] = float(splitted_line[tongue.RATING_INDEX])  # ratings are incremented in steps of 0.5
			splitted_line[tongue.TIME_INDEX] = int(splitted_line[tongue.TIME_INDEX])
			data.append(splitted_line)
		return data  # data[j] is list representing line j. index 0 in list is user id, index 1 is item id, index 2 is rating, index 3 is time

	def create_ratings_matrix(self):
		ratings_matrix = lil_matrix((self.__data.get_num_of_users(), self.__data.get_num_of_items()))
		for current_user_id, current_item_id, current_rating, _ in self.__data:
			ratings_matrix[current_user_id, current_item_id] = current_rating
		return csr_matrix(ratings_matrix)


if __name__ == "__main__":
	parser = Parser(file_path = 'ml-100k/u.data')
	ratings_matrix = parser.create_ratings_matrix()
	print("rating matrix: %s" % ratings_matrix)
