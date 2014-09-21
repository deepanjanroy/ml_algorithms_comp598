import dataparser
import sys
from numpy import *
from gradientdescent import GradientDescentRunner

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

    def get_gradient(self):
        X = self.X
        Y = self.Y
        XTX = X.T.dot(X)
        return lambda w: 2 * (XTX.dot(w) - X.T.dot(Y))



    def training_error(self):
        errors = (self.predict_multiple(self.X) - self.Y)
        return linalg.norm(errors) / self.X.shape[0]

    def gradient_descent(self, alpha=1e-11, epsilon=1e-5):
        self.weights = self.gradient_descent_once(alpha, epsilon)


    def test_error(self, samples, labels):
        shaped_input_list = array([ input + [1] for input in samples])
        predictions = (shaped_input_list).dot(self.weights)
        errors = predictions - array(labels)
        return errors.sum() / shaped_input_list.shape[0]



def calculate_test_error(regressor, test_data, test_labels):
    predictions = array(regressor.predict_multiple(test_data))
    errors = predictions - array(test_labels)
    return linalg.norm(errors) / errors.shape[0]

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


# Not sure how to properly test gradient descent
def test_gradient_descent(samples, test_data, scikit_pred):
    reg = LinearRegressor(samples, labels)
    gr = GradientDescentRunner(reg.get_gradient(), len(samples[0]) + 1)
    weights = gr.run_once()
    reg.weights = weights
    p = reg.predict(test_data)
    print "Prediction from gradient descent: ", p


# Test error
def test_regression_test_error(train_data, train_label, test_data, test_label):
    reg = LinearRegressor(train_data, train_label)
    reg.train()
    print "Test error: ", calculate_test_error(reg, test_data, test_labels)

if __name__ == "__main__":
    from sklearn import linear_model
    from numpy import testing

    samples, labels = dataparser.parse_dataset("Example.arff.txt")

    half_index = len(samples) / 2

    train_dataset = samples[:half_index]
    train_labels = labels[:half_index]

    test_data = samples[0]
    test_dataset = samples[half_index:]
    test_labels = labels[half_index:]

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
    test_regression_test_error(train_dataset, train_labels, test_dataset, test_labels)

