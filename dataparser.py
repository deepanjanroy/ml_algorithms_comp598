import csv

def normalize_data(self):
	num_features = len(self.samples[0])
	for i in xrange(num_features):
		all_values= [r[i] for r in self.samples]
		max_value = max(all_values, key=lambda x: abs(x))
		print max_value
		for row in self.samples:
			# try:
			# 	assert type(row[i]) == float
			# except:
			# 	print "Shit just hit the fan"
			# 	print row[i]
			# 	print type(row[i])
			# 	raise
			row[i] = float(row[i]) / max_value



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