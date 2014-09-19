from sklearn import linear_model
import csv

total_samples = 0
with open('Example.arff.txt') as f:
	csv_file = csv.reader(f)
	data = []
	label = []
	for row in csv_file:
		total_samples += 1
		data.append(row[:-1])
		label.append(row[-1])


clf = linear_model.LogisticRegression()
train_set = (data[:total_samples/2], label[:total_samples/2])
test_set = (data[total_samples/2:], label[total_samples/2:])

clf.fit(*train_set)

c