import unittest
import dataparser
from numpy import *
from numpy import testing
from logistic_regression import LogisticRegressor
from gradientdescent import GradientDescentRunner
from scales import Scale

class TestLogisticRegression(unittest.TestCase):
    samples, labels = dataparser.parse_dataset("/Users/droy/Downloads/dataset.csv")
    split_index = int(len(samples) * 0.75)

    train_dataset = samples[:split_index]
    train_labels = labels[:split_index]

    test_dataset = samples[split_index:]
    test_labels = labels[split_index:]

    test_data = samples[0]
    def setUp(self):
        self.logreg = LogisticRegressor(self.train_dataset, self.train_labels)

    def test_gradient(self):
        """Just testing that it doesn't throw any errors"""
        g = self.logreg.get_gradient()
        print g(ones(len(self.samples[0]) + 1))

    def test_gradient_descent(self):
        scale = Scale.scale_from_data(self.test_dataset)
        # self.logreg = LogisticRegressor(self.train_dataset, self.train_labels, scale=scale)

        self.logreg = LogisticRegressor(self.train_dataset, self.train_labels)

        gr = GradientDescentRunner(self.logreg.get_gradient(), len(self.samples[0]) + 1,
                                   self.logreg.get_error_function(), alpha=1e-8, max_iter=300)
        weights = gr.run()
        self.logreg.weights = weights

        mismatches = 0
        predictions = [self.logreg.predict(test_data) for test_data in self.test_dataset]
        print "All predictions: ", predictions

        for test_data, label in zip(self.test_dataset, self.test_labels):
            prediction = self.logreg.predict(test_data)
            if prediction != label:
                mismatches +=1
                # print "Mismatch! Predicted ", prediction, ", True ", label

        total = len(self.test_dataset)
        print "total data: ", total
        print "total mismatch: ", mismatches
        print "percentage success: ", (100 - float(mismatches) / float(total) * 100), "%"

    def test_sckikit_performance(self):
        from sklearn import linear_model
        clf = linear_model.LogisticRegression()
        clf.fit(self.train_dataset, self.train_labels)
        predictions = clf.predict(self.test_dataset)
        print "Scikit predictions: ", predictions

        mismatches = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for test_data, label in zip(self.test_dataset, self.test_labels):
            prediction = clf.predict(test_data)

            if prediction != label:
                mismatches += 1

            if prediction == 1 and label == 0:
                false_positives += 1

            if prediction == 0 and label == 1:
                false_negatives += 1

            if prediction == 1 and label == 1:
                true_positives += 1

            if prediction == 0 and label == 0:
                true_negatives += 1

                # print "Mismatch! Predicted ", prediction, ", True ", label

        total = len(self.test_dataset)
        print "True positives: ", true_positives
        print "True negatives: ", true_negatives
        print "False negatives: ", false_negatives
        print "False positives: ", false_positives

        print "F1 metric: ", 2 * float(true_positives) \
                             / float(2 * true_positives + false_positives + false_negatives)

        print "MCC: ", float(true_positives * true_negatives - false_positives * false_negatives) \
                    / math.sqrt((true_positives + false_positives) * (true_positives + false_negatives) \
                                * (true_negatives + false_positives) * (true_negatives + false_negatives))

        print "total data: ", total
        print "total mismatch: ", mismatches
        print "percentage success: ", (100 - float(mismatches) / float(total) * 100), "%"


    def test_lets_just_look_at_the_outputs(self):
        gr = GradientDescentRunner(self.logreg.get_gradient(), len(self.samples[0]) + 1,
                                   self.logreg.get_error_function(), alpha=1e-8, max_iter=300)

        _, weights = gr.run_once()
        self.logreg.weights = weights

        predictions = [self.logreg.get_probability(d) for d in self.train_dataset]
        import pprint
        pprint.pprint(zip(predictions, self.train_labels))

if __name__ == "__main__":
    unittest.main()