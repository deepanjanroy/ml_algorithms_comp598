import numpy as np
from scales import Scale

sigmoid = lambda x: 1 / (1 + np.exp(-x))


### debug
# import warnings
# warnings.filterwarnings("error")


class LogisticRegressor(object):
    threshold = 0.5

    def __init__(self, train_data, train_labels, weights=None, scale=None):

        # Adding 1 at the end for the constant term
        list_X = [r + [1] for r in train_data]

        if scale is None:
            self.scale = Scale.unit_scale(len(list_X[0]))
        else:
            self.scale = scale.add_unit_elements(1)

        self.weights = weights

        self.X = self.scale(np.array(list_X))
        self.Y = np.array(train_labels)

    def get_gradient(self):
        X = self.X
        Y = self.Y

        def gradient(w):
            sum = np.zeros(w.shape)
            for i in xrange(len(X)):
                sum += X[i].dot( Y[i] - sigmoid(w.T.dot(X[i])) )
            return -sum

        return gradient

    def get_error_function(self):
        X = self.X
        Y = self.Y

        # Cross entropy error function
        def error_fn(w):
            sum = 0

            for i in xrange(len(X)):
                sigmoid_wdotx = sigmoid(w.T.dot(X[i]))
                # Massive hack. Why sigmoid(wdotx) is ever 0 or 1 is a mystery to me
                if sigmoid_wdotx == 1:
                    sigmoid_wdotx = 0.9999999999
                elif sigmoid_wdotx == 0:
                    sigmoid_wdotx = 0.0000000001

                sum += Y[i] * np.log(sigmoid_wdotx) \
                            + (1 - Y[i] * np.log(1 - sigmoid_wdotx))


            return -sum

        return error_fn


    def predict(self, input):
        input = self.scale(np.array(input + [1]))
        probability = sigmoid(self.weights.dot(input))

        if probability > self.threshold:
            return 1
        else:
            return 0

    def get_probability(self, input):
        input = self.scale(np.array(input + [1]))
        return sigmoid(self.weights.dot(input))

    def predict_multiple(self, inp_list):
        return [self.predict(input) for input in inp_list]

