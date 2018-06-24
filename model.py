import os
import csv

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

image_labels = []
image_names = []
image_data = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		for i in range(3):
			image_names.append('./data/' + line[i].strip())
		steering_value = float(line[3])
		image_labels.append(steering_value)
		image_labels.append(steering_value + 0.2)
		image_labels.append(steering_value - 0.2) 

print(image_names[0])
#testing for a few images
for i in range(50):
	image_data.append(np.array(Image.open(image_names[i])))

print(image_data[0].shape)
