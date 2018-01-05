# -*- coding: utf-8 -*-
# encoding=utf8

import os, sys
import csv
import cv2
import pandas as pd
import numpy as np
import sklearn


# Returns the lines from a driving logs

def read_logs(data_path, skip=False):

    '''
    Reading logs from the input data located in data_path:
    Returns an array of observation with steering angle information from csv
    '''

    lines = []

    with open(os.path.join(data_path, 'driving_log.csv')) as csvFile:
        reader = csv.reader(csvFile)
        if skip:
            next(reader, None)
        for line in reader:
            lines.append(line)
    
    return lines


# Process the logs to return paths and steering measurements

def read_images(data_path):

    lines = read_logs(data_path)

    col_names = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

    df_lines = pd.DataFrame(lines, columns=col_names)

    leftTotal = list(df_lines.left)
    centerTotal = list(df_lines.center)
    rightTotal = list(df_lines.right)
    measurements_str = list(df_lines["steering"])
    measurements = [float(x) for x in measurements_str]

    return leftTotal, centerTotal, rightTotal, measurements


# Combine measurement data

def combine_measurements(left_paths, center_paths, right_paths, measurement, correction):

    image_paths = left_paths + center_paths + right_paths

    measurements = [x + correction for x in measurement] + measurement + [x - correction for x in measurement]

    return image_paths, measurements



# Generates data in batches to be fed into the keras fit_generator object

def generate_data(examples, batch_size=32):

    ''' 
    Arguments
        examples: the array of observation data that is to be split into batches and read into image arrays
        batch_size: batches of images to be fed to Keras model
    Returns
        X: image array in batches as a list
        y: steering angle list 
    '''

    num_examples = len(examples)

    while True: # Loop forever so the generator never terminates

        examples = sklearn.utils.shuffle(examples)
        
        for offset in range(0, num_examples, batch_size):
            
            batch_samples = examples[offset:offset+batch_size]

            images = []
            angles = []

            for image_path, measurement in batch_samples:

                raw_image = cv2.imread(image_path)

                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)

            yield sklearn.utils.shuffle(inputs, outputs)



