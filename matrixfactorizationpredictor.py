import numpy as np
import pickle

class MatrixFactorizationPredictor():
	def __init__(self, K=15, alpha=0.001, beta=0, iterations=10):
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.iterations = iterations

	def train(self, training_set_file_path):
		with open(training_set_file_path, 'rb') as training_set_file:
			R = pickle.load(training_set_file)
			self.R = R
			self.num_users, self.num_items = R.shape

			# Initialize user and item latent feature matrice
			self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
			self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

			# Initialize the biases
			self.b_u = np.zeros(self.num_users)
			self.b_i = np.zeros(self.num_items)
			self.b = np.mean(self.R[self.R.nonzero()])

			# Create a list of training samples
			rows_non_zero, cols_non_zero = self.R.nonzero()
			self.samples = [(rows_non_zero[i], cols_non_zero[i], self.R[rows_non_zero[i], cols_non_zero[i]]) for i in range(len(rows_non_zero))]

			# Perform stochastic gradient descent for number of iterations
			for i in range(self.iterations):
				self.sgd()

	def sgd(self):
		for i, j, r in self.samples:
			# Computer prediction and error
			prediction = self.get_rating(i, j)
			e = (r - prediction)

			# Update biases
			self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
			self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

			# Update user and item latent feature matrices
			self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
			self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

	def get_rating(self, i, j):
		prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
		return prediction

	def predict(self, testing_set_file_path):
		with open(testing_set_file_path, 'rb') as testing_set_file:
			testing_set = pickle.load(testing_set_file)
			rows_non_zero, cols_non_zero = testing_set.nonzero()
			mae = 0
			for i in range(len(rows_non_zero)):
				user, item = rows_non_zero[i], cols_non_zero[i]
				actual_rating = testing_set[user, item]
				predicted_rating = self.get_rating(user, item)
				mae += abs(actual_rating - predicted_rating)
			mae /= len(rows_non_zero)
			return mae