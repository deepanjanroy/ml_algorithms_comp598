import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class LogisticRegressor(object):
    def __init__(self, train_data, train_labels):
        list_X = [r + [1] for r in train_data]
        self.X = array(list_X)
        self.Y = array(train_labels)

    def get_gradient(self):
        X = self.X
        Y = self.Y

        def gradient(w):
            sum = np.zeros(w.shape)
            for i in xrange(len(X)):
                sum += X[i].dot( Y[i] - sigmoid(w.T.dot(X[i])) )
            return sum

        return gradient


#Tests

if __name__ == "__main__":
    from sklearn import linear_model
    from numpy import testing
    from numpy import *
    import dataparser

    samples, labels = dataparser.parse_dataset("Example.arff.txt")

    half_index = len(samples) / 2

    train_dataset = samples[:half_index]
    train_labels = labels[:half_index]

    test_data = samples[0]
    test_dataset = samples[half_index:]
    test_labels = labels[half_index:]

    logreg = LogisticRegressor(train_dataset, train_labels)
    g = logreg.get_gradient()
    print g(np.ones(len(samples[0]) + 1))
