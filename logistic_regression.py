import numpy as np

class LogisticRegressor(object):
	weights = None
    normalizers = None


class Scale(object):
    """Used for scaling data or test vectors.

    """
    def __init__(self, length=1):
        """
        :return: Scale object
        """
        self.factors = ones((length,))

    def scale_vector(self):
        """
        Does something
        :return:
        """
        pass



