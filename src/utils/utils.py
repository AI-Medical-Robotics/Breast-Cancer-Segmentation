# common
import os
import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf

# Data
#from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 

from keras import layers
from keras import models

# Callbacks 
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tf_explain.callbacks.grad_cam import GradCAM
#https://tf-explain.readthedocs.io/en/latest/usage.html

# Metrics
from keras.metrics import MeanIoU

SIZE = 256

def loadOneImage(image, SIZE):
    return np.round(tf.image.resize(img_to_array(load_img(image))/255.,(SIZE, SIZE)),4)

def loadImages(path, SIZE, mask=False, trim=None):
    if trim is not None:
        path = path[:trim]
    if mask:
        images = np.zeros(shape=(len(path), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(path), SIZE, SIZE, 3))
    
    for i,image in enumerate(path):
        img = loadOneImage(image, SIZE)
        if mask:
            images[i] = img[:,:,:1]
        else:
            images[i] = img
    
    return images

def showImage(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def showMask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')