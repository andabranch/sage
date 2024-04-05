from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import os
import collections
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import contextlib
import csv

import larq as lq

IMG_RESIZE = (64, 64)
OUTPUT_PATH = 'output/best/'
nr_img = 6020
nr_img_start = 0
dataset_path = "perturb_3/"
img_name = "00001.png"
img_class = 13

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}


def create_file():
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with column names from the dictionary values
        writer.writerow(classes[key] for key in sorted(classes.keys()))


def get_perturb_ds():
    data = []
    y_data = []
    for i in range(nr_img_start, nr_img):
        image = Image.open(dataset_path + str(i) + "_" + img_name)
        image = image.resize(IMG_RESIZE)
        data.append(np.array(image))
        y_data.append(img_class)

    X_test = np.array(data)
    y_test = np.array(y_data)
    return X_test, y_test


def get_perturb_ds1():
    data = []
    y_data = []
    
    paths = []

    # Iterate over files in the folder
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)

        # Check if the file is an image (example: jpg, png)
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):

            image = Image.open(file_path)
            image = image.resize(IMG_RESIZE)
            data.append(np.array(image))
            y_data.append(img_class)
            paths.append(file_path)

    X_test = np.array(data)
    y_test = np.array(y_data)
    return X_test, y_test, paths


def get_n_save_accuracy(X_test, y_test, test_name):
    model = tf.keras.models.load_model(
        OUTPUT_PATH + 'models/' + test_name + '.h5')
    pred = model.predict(X_test)
    y_pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test, np.argmax(pred, axis=1))

    # Set the print options to display all values without truncation
    np.set_printoptions(threshold=np.inf)
    print(pred)
    np.savetxt(dataset_path + 'output.csv', pred, delimiter=',')

    print(' dataset accuracy: ' + str(acc))
    return y_pred


X_test, y_test, paths = get_perturb_ds1()
test_name = '3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30'
y_pred = get_n_save_accuracy(X_test, y_test, test_name)

print(y_pred)
for i in range(len(y_pred)):
    if(y_pred[i] != img_class):
        print(paths[i] + ": " + str(i))

# create_file()
