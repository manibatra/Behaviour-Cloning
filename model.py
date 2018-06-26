import os
import csv

import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

image_labels = []
image_names = []
image_data = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		for i in range(3):
			image_names.append('./data/IMG/' + line[i].strip().split('/')[-1])
		steering_value = float(line[3])
		image_labels.append(steering_value)
		image_labels.append(steering_value + 0.1)
		image_labels.append(steering_value - 0.1) 

print(image_names[0])
#testing for a few images
for i in range(len(image_names)):
	image_data.append(mpimg.imread(image_names[i]))

image_data = np.array(image_data)
image_labels = np.array(image_labels)
shape = image_data[0].shape


#create the model : Comma.ai
model = Sequential()
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=shape))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(image_data, image_labels, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')

