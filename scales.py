import numpy as np


class Scale(object):
    def __init__(self, factors):
        self.factors = factors

    @staticmethod
    def scale_from_data(data):
        factors = []
        data_vector_dimension = len(data[0])
        for i in xrange(data_vector_dimension):
            max_in_col = max([abs(row[i]) for row in data])
            factors.append(max_in_col)

        return Scale(np.array(factors))

    @staticmethod
    def unit_scale(dim):
        factors = np.ones(dim)
        return Scale(factors)
