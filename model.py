import os
import csv

import matplotlib.image as mpimg
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

image_labels = []
image_names = []
image_data = []

with open('./examples/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		for i in range(3):
			image_names.append('./examples/data/IMG/' + line[i].strip().split('/')[-1])
		steering_value = float(line[3])
		image_labels.append(steering_value)
		image_labels.append(steering_value + 0.2)
		image_labels.append(steering_value - 0.2) 

#reading images
for i in range(len(image_names)):
	image_data.append(mpimg.imread(image_names[i]))

#flip images horizontally
for i in range(len(image_data)):
	image_data.append(cv2.flip(image_data[i], 0))
	image_labels.append(image_labels[i] * -1)

image_data = np.array(image_data)
image_labels = np.array(image_labels)
shape = image_data[0].shape



#create the model : Comma.ai
def get_comma_ai_model():
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
	return model

#model based on the nvidia paper: 
#http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def get_nvidia_model():
	model = Sequential()
	model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=shape))
	model.add(Lambda(lambda x: x/127.5 - 1.))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(64, 3, 3, activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
#	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

model = get_nvidia_model()
model.compile(loss='mse', optimizer='adam')
model.fit(image_data, image_labels, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_nvidia.h5')

