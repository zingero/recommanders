import tongue
from parsers.abstractparser import AbstractParser


class Parser100K(AbstractParser):
	def _parse_line(self, line):
		splitted_line = line.split('\t')  # get rid of \t and neatly organize different columns in lists
		splitted_line[tongue.USER_ID_INDEX] = int(splitted_line[tongue.USER_ID_INDEX]) - 1  # make user ids start from 0
		splitted_line[tongue.ITEM_ID_INDEX] = int(splitted_line[tongue.ITEM_ID_INDEX]) - 1  # make item ids start from 0
		splitted_line[tongue.RATING_INDEX] = int(splitted_line[tongue.RATING_INDEX])
		splitted_line[tongue.TIME_INDEX] = int(splitted_line[tongue.TIME_INDEX])
		return splitted_line
