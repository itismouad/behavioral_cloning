# -*- coding: utf-8 -*-
# encoding=utf8

import os, sys
import csv
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


# Setting up useful paths

project_path = os.path.join(os.environ['HOME'], "sdcnd/behavioral_cloning")
utlis_path = os.path.join(project_path, "utils")
data_path = os.path.join(project_path, "data")

# Loading custom package for data mungling

sys.path.insert(0, utlis_path)
from data_mungling import *


# Load and generate the data

left_paths, center_paths, right_paths, measurements = read_images(data_path)

image_paths, measurements = combine_measurements(left_paths, center_paths, right_paths, measurements, correction=0.2)

data = list(zip(image_paths, measurements))


# Splitting samples and creating generators

training_data, test_data = train_test_split(data, test_size=0.2)

train_generator = generate_data(training_data, batch_size=32)
validation_generator = generate_data(test_data, batch_size=32)


# Design and train the model based on nvidia autonomous car team

def model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Model setup

model = model()


# Defining how the model will compile

model.compile(loss='mse', optimizer='adam')


# Fitting model

history_object = model.fit_generator(
	train_generator, samples_per_epoch= len(training_data),
	validation_data=test_generator, nb_val_samples=len(test_data),
	nb_epoch=5,
	verbose=1
	)

# Saving model

model.save(os.path.join(project_path, 'model.h5'))

