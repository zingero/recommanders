import os
import pickle
import numpy as np

import tongue


class UniformPredictor(object):
    def predict(self, testing_set_file_name, possible_values):
        with open(os.path.join(tongue.PARSED_DATA_PATH, testing_set_file_name), 'rb') as testing_set_file:
            testing_set = pickle.load(testing_set_file)
            rows_non_zero, cols_non_zero = testing_set.nonzero()
            mae = 0
            for i in range(len(rows_non_zero)):
                current_rating = testing_set[rows_non_zero[i], cols_non_zero[i]]
                current_prediction = possible_values[np.random.randint(0, len(possible_values))]
                mae += abs(current_rating - current_prediction) / len(rows_non_zero)
            return mae
