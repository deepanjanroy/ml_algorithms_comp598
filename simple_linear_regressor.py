import dataparser
import sys
from numpy import *

inv = linalg.inv


class LinearRegressor(object):
    weights = None
    normalizers = None

    def __init__(self, samples, labels):
        list_X = [r + [1] for r in samples]
        self.X = array(list_X)
        self.Y = array(labels)

    def train(self):
        # from IPython import embed; embed()
        X = self.X
        Y = self.Y

        XT = X.transpose()
        XTX = XT.dot(X)
        self.weights = inv(XTX).dot(XT).dot(Y)

    def normalize(self):
        max_values = [max([abs(el) for el in column]) for column in self.X.transpose()]
        self.normalizers = array([max_values])

    def predict(self, input):
        shaped_input = array([input + [1]])
        if self.normalizers is not None:
            shaped_input = shaped_input / self.normalizers

        return shaped_input.dot(self.weights)[0]

    def predict_multiple(self, input_list):
        shaped_input_list = array([ input + [1] for input in input_list])
        return (shaped_input_list).dot(self.weights).tolist()

    def gradient(self):
        return self._gradient(self.weights, self.X, self.Y, self.X.T.dot(self.X))


    def _gradient(self, w, X, Y, XTX):
        XTX = X.T.dot(X)
        return 2 * (XTX.dot(w) - X.T.dot(Y))

    def gradient_descent_once(self, alpha, epsilon):
        print "Running gradient descent... This may take a while"
        XTX = self.X.T.dot(self.X)
        w_prev = random.rand(self.X.shape[1],)
        w_next = w_prev - alpha * self._gradient(w_prev, self.X, self.Y, XTX)

        iterations = 0
        while linalg.norm(w_next - w_prev) > epsilon:
            w_prev = w_next
            w_next = w_prev - alpha * self._gradient(w_prev, self.X, self.Y, XTX)

            # print "Change: ", linalg.norm(w_next - w_prev)
            iterations += 1
            if iterations % 1e3 == 0:
                sys.stdout.write("Iterations complete: %d   \r" % (iterations,) )
                sys.stdout.flush()

            # print "weights: ", self.weights
            # print "w_prev: ", w_prev

        return w_next

    def training_error(self):
        errors = (self.predict_multiple(self.X) - self.Y) ** 2
        return errors.sum() / self.X.shape[0]

    def gradient_descent(self, alpha=1e-11, epsilon=1e-5):
        self.weights = self.gradient_descent_once(alpha, epsilon)



## TESTS

def show_normalization_sucks(samples, test_data):
    print "Showing strange behaviors with normalization:"
    for i in xrange(len(samples[0])):
        print "Normalizing column {0}".format(i)

        max_in_col = max([abs(r[i]) for r in samples])

        n_samples = [r[:i] + [float(r[i])/max_in_col] + r[i+1:] for r in samples]
        new_reg = LinearRegressor(n_samples, labels)
        new_reg.train()
        p = new_reg.predict(test_data)
        print "Prediction: " + str(p)

        scikit_clf = linear_model.LinearRegression()
        scikit_clf.fit(n_samples, labels)
        p = scikit_clf.predict(test_data)
        print "Scikit prediction: " + str(p)

def test_gradient_descent(samples, test_data, scikit_pred):
    reg = LinearRegressor(samples, labels)
    reg.gradient_descent(epsilon=1)
    print "Weights: ", reg.weights
    print "prediction: ", reg.predict(test_data)
    print "Scikit prediction: ", scikit_pred

if __name__ == "__main__":
    from sklearn import linear_model
    from numpy import testing

    samples, labels = dataparser.parse_dataset("Example.arff.txt")

    half_index = len(samples) / 2

    train_dataset = samples[:half_index]
    train_labels = labels[:half_index]

    test_data = samples[0]
    test_dataset = samples[half_index:]

    reg = LinearRegressor(train_dataset, train_labels)
    reg.train()
    print "Training error: ", reg.training_error()
    our_single_prediction = reg.predict(test_data)
    our_predictions = reg.predict_multiple(test_dataset)

    clf = linear_model.LinearRegression()
    clf.fit(train_dataset, train_labels)
    scikit_single_prediciton = clf.predict(test_data)
    scikit_multiple_prediction = clf.predict(test_dataset)

    testing.assert_almost_equal(our_single_prediction, scikit_single_prediciton)
    testing.assert_array_almost_equal(our_predictions, scikit_multiple_prediction)

    ## Testing gradients
    test_gradient_descent(samples, test_data, scikit_single_prediciton)