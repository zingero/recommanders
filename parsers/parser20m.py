import tongue
from parsers.abstractparser import AbstractParser


class Parser20M(AbstractParser):
	def _parse_line(self, line):
		if line == "userId,movieId,rating,timestamp":
			return None
		splitted_line = line.split(',')  # get rid of , and neatly organize different columns in lists
		splitted_line[tongue.USER_ID_INDEX] = int(splitted_line[tongue.USER_ID_INDEX]) - 1  # make user ids start from 0
		splitted_line[tongue.ITEM_ID_INDEX] = int(splitted_line[tongue.ITEM_ID_INDEX]) - 1  # make item ids start from 0
		splitted_line[tongue.RATING_INDEX] = int(float(splitted_line[tongue.RATING_INDEX]) * 2)  # scaling 0-5 with step of 0.5 to 0-10 with step of 1
		splitted_line[tongue.TIME_INDEX] = int(splitted_line[tongue.TIME_INDEX])
		return splitted_line
