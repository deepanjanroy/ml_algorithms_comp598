import numpy as np


class Scale(object):
    """Scales a vector with vector factors"""

    def __init__(self, factors):
        """
        :param factors: numpy array. NOT a regular array
        :return: scale
        """
        self.factors = factors

    @staticmethod
    def scale_from_data(data):
        factors = []
        data_vector_dimension = len(data[0])
        for i in xrange(data_vector_dimension):
            max_in_col = max([abs(row[i]) for row in data])
            factors.append(1.0/float(max_in_col))

        return Scale(np.array(factors, dtype=np.float64))

    @staticmethod
    def unit_scale(dim):
        factors = np.ones(dim)
        return Scale(factors)

    def scale(self, arg):
        return np.array(arg) * (self.factors)

    def add_unit_elements(self, num_unit_elms):
        """Adds a bunch of 1s at the end of factors.

        Useful when we one to strech the scale a bit just to make
        matrix multiplications work.
        """
        return Scale(np.concatenate([self.factors, np.ones(num_unit_elms)]))

    def __mul__(self, other):
        return Scale(self.factors * other)

    def __call__(self, *args, **kwargs):
        return self.scale(*args, **kwargs)