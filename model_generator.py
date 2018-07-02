import os
import csv

import matplotlib.image as mpimg
import cv2
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split
import sklearn


image_labels = []
image_names = []
image_data = []

with open('./track_2_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		for i in range(3):
			image_names.append('./track_2_data/IMG/' + line[i].strip().split('/')[-1])
		steering_value = float(line[3])
		image_labels.append(steering_value)
		image_labels.append(steering_value + 0.2)
		image_labels.append(steering_value - 0.2) 

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

train_samples, validation_samples, train_labels, validation_labels  = train_test_split(image_names, image_labels, test_size=0.2)



def generator(samples, labels, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]
			batch_angles = labels[offset:offset + batch_size]

			images = []
			angles = []

			for i in range(len(batch_samples)):
				image = mpimg.imread(batch_samples[i])
				#print(image.shape)
				c1, c2, c3 = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
				image = np.expand_dims(c1, axis=3)
				#print(image.shape)
				angle = batch_angles[i]

				images.append(image)
				angles.append(angle)

				image = np.expand_dims(cv2.flip(image, 0), axis=3)
				angle *= -1
				#print("image after flip", image.shape)
				images.append(image)
				angles.append(angle)
			
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)	


train_generator = generator(train_samples, train_labels, batch_size=128)
validation_generator = generator(validation_samples, validation_labels, batch_size=128)


				

#create the model : Comma.ai
def get_comma_ai_model():
	model = Sequential()
	model.add(Cropping2D(cropping=((80, 20), (0, 0)), input_shape=shape))
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
	model.add(Cropping2D(cropping=((80, 20), (0, 0)), input_shape=(160, 320, 1)))
	model.add(Lambda(lambda x: x/127.5 - 1.))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
#	model.add(MaxPooling2D())
	model.add(Convolution2D(64, 3, 3, activation='relu'))
#	model.add(MaxPooling2D())
#	model.add(Convolution2D(64, 3, 3, activation='relu'))
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
#model = load_model("model_nvidia.h5")
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model_nvidia_track2.h5')

