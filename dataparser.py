import csv
import itertools


def parse_dataset(filename):
    """
    Parses a csv file given by filename. Assumes the last entry in each row is
    a label. Returns a tuple of lists: (data, label)
    """

    samples = []
    labels = []
    with open(filename) as f:
        csv_file = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_file:
            samples.append(row[:-1])
            labels.append(row[-1])

    return (samples, labels)

# factor_difference: 0 to 1
def get_sampled_dataset(vectors, labels, factor_difference):
    """We always have more negative than positive.
    This method undersamples the negative vectors
    """

    positive_vectors = []
    negative_vectors = []
    for (vector, label) in itertools.izip(vectors, labels):
        if label == 1:
            positive_vectors.append(vector)
        else:
            negative_vectors.append(vector)

    step_size = float(len(negative_vectors)) / (factor_difference * len(positive_vectors))

    sampled_negatives = []
    i = 0
    while (i * step_size < len(negative_vectors)):
        sampled_negatives.append(negative_vectors[int(step_size * i)])
        i += 1

    sampled_vectors = positive_vectors + sampled_negatives
    sampled_labels = [1 for v in positive_vectors] + [0 for v in sampled_negatives]

    print "Positive vectors: ", len(positive_vectors)
    print "Negative vectors: ", len(sampled_negatives)
    return sampled_vectors, sampled_labels

def split_train_and_test(positives, negatives, percentage_in_train):
    pos_index = int(len(positives) * percentage_in_train)
    neg_index = int(len(negatives) * percentage_in_train)

    train_positives_samples = positives[:pos_index]
    train_positive_labels = [ 1 for d in train_positives_samples ]

    test_positive_samples = positives[pos_index:]
    test_positive_labels = [ 1 for d in test_positive_samples ]

    train_negative_samples = negatives[:neg_index]
    train_negative_labels = [ 0 for d in train_negative_samples ]

    test_negative_samples = negatives[neg_index:]
    test_negative_labels = [ 0 for d in test_negative_samples ]

    train_samples = train_positives_samples + train_negative_samples
    train_labels = train_positive_labels + train_negative_labels

    test_samples = test_positive_samples + test_negative_samples
    test_labels = test_positive_labels + test_negative_labels

    return {
                "train": (train_samples, train_labels),
                "test": (test_samples, test_labels)
            }


