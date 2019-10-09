import abc

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

from data import Data


class AbstractParser(object):
	__meta__ = abc.ABCMeta

	def __init__(self, file_path):
		self._data = self.__parse_file(file_path)

	def __parse_file(self, file_path):
		with open(file_path, "r") as f:
			data_file = f.readlines()
		data = Data()
		i = 0
		for line in data_file:
			i += 1
			parsed_line = self._parse_line(line.strip())
			if parsed_line:
				data.append(parsed_line)
			if i % 10000 == 0:
				print("parsed %d lines" % i)
		return data  # data[j] is list representing line j. index 0 in list is user id, index 1 is item id, index 2 is rating, index 3 is time

	def create_ratings_matrix(self):
		ratings_matrix = lil_matrix((self._data.get_num_of_users(), self._data.get_num_of_items()))
		for current_user_id, current_item_id, current_rating, _ in self._data:
			ratings_matrix[current_user_id, current_item_id] = current_rating
		return csr_matrix(ratings_matrix)

	@abc.abstractmethod
	def _parse_line(self, line):
		pass
