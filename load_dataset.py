# import

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import utils_paths
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import random
import pickle
import cv2
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint


def input_data_train():
    # load data
    print("------Begin load data------")
    data = []
    data_noise = []

    
    imagePaths1 = sorted(list(utils_paths.list_images(
        'E:\\train\\picture')))

    imagePaths2 = sorted(list(utils_paths.list_images(
        'E:\\train\\noise')))
    # print(imagePaths2)
    
    for imagePath in imagePaths1:
        #print(imagePath)
        
        image = cv2.imread(imagePath)
        # print(image.shape)
        (r, g, b) = cv2.split(image)
        image = cv2.merge([b, g, r])

        image = cv2.resize(image, (128, 128))
        # plt.imshow(image)
        # plt.show()
        data.append(image)
    for imagePath in imagePaths2:
       
        image = cv2.imread(imagePath)
        # print(image.shape)
        (r, g, b) = cv2.split(image)
        image = cv2.merge([b, g, r])

        image = cv2.resize(image, (128, 128))
        data_noise.append(image)
        

    # scale

    data = (np.array(data, dtype="float32") - 127.5) / 127.5
    data_noise = (np.array(data_noise, dtype="float32") - 127.5) / 127.5
    # print(labels)
    

    # print(trainY)
    # one-hot encoding

    #trainY = lb.fit_transform(trainY)
    #testY = lb.transform(testY)
    return data, data_noise
#data, data_noise = input_data_train()

