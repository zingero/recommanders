import tongue


class Data(list):
	def __init__(self):
		super(Data).__init__()
		self.__num_of_users = None
		self.__num_of_items = None

	def get_num_of_users(self):
		if not self.__num_of_users:
			self.__num_of_users = max([item[tongue.USER_ID_INDEX] for item in self]) + 1  # +1 since they start from 0
		return self.__num_of_users

	def get_num_of_items(self):
		if not self.__num_of_items:
			self.__num_of_items = max([item[tongue.ITEM_ID_INDEX] for item in self]) + 1  # +1 since they start from 0
		return self.__num_of_items
