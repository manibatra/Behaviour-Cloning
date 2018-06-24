import os
import csv

image_labels = []
image_data = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		for i in range(3):
			image_data.append('data/' + line[i])
		steering_value = float(line[3])
		image_labels.append(steering_value)
		image_labels.append(steering_value + 0.2)
		image_labels.append(steering_value - 0.2)


print(len(image_labels), len(image_data))
