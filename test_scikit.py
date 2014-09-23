import dataparser
from sklearn import linear_model
import unittest
import numpy as np

data_file_path = "/Users/droy/Downloads/dataset.csv"

def calculate_MCC_metric(true_positives, true_negatives, false_positives, false_negatives):
    return float(true_positives * true_negatives - false_positives * false_negatives) \
            / np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) \
              * (true_negatives + false_positives) * (true_negatives + false_negatives))

def calculate_F1_metric(true_positives, true_negatives, false_positives, false_negatives):
    return 2 * float(true_positives) \
           / float(2 * true_positives + false_positives + false_negatives)

class TestScikit(unittest.TestCase):

    vectors, labels = dataparser.parse_dataset(data_file_path)

    split_index = int(0.75 * len(vectors))
    train_vectors = vectors[:split_index]
    test_vectors = vectors[split_index:]

    train_labels = labels[:split_index]
    test_labels = labels[split_index:]


    def get_sckikit_performance(self, factor):
        sampled_vectors, sampled_labels = dataparser.get_sampled_dataset(self.train_vectors, self.train_labels, factor)

        clf = linear_model.LogisticRegression()
        clf.fit(sampled_vectors, sampled_labels)
        predictions = clf.predict(self.test_vectors)
        print "Scikit predictions: ", predictions

        mismatches = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for test_data, label in zip(self.test_vectors, self.test_labels):
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

        total = len(self.test_vectors)
        # print "True positives: ", true_positives
        # print "True negatives: ", true_negatives
        # print "False negatives: ", false_negatives
        # print "False positives: ", false_positives

        # print "F1 metric: ", calculate_F1_metric(true_positives, true_negatives, false_positives, false_negatives)
        f1_metric = calculate_F1_metric(true_positives, true_negatives, false_positives, false_negatives)

        MCC_metric = calculate_MCC_metric(true_positives, true_negatives, false_positives, false_negatives)
        # print "MCC: ", MCC_metric
        #
        # print "total data: ", total
        # print "total mismatch: ", mismatches
        # print "percentage success: ", (100 - float(mismatches) / float(total) * 100), "%"

        return {"MCC_metric": MCC_metric,
                "F1_metric" : f1_metric}




    def test_factors(self):
        factors = [float(x) / 10 for x in xrange(5, 41, 1)]
        results = []
        for factor in factors:
            print "Using factor ", factor
            score = self.get_sckikit_performance(factor)["F1_metric"]
            results.append((factor, score))

        print results



if __name__ == "__main__":
    unittest.main()