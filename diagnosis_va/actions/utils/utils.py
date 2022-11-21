# common
# import os
# import tensorflow.keras
import numpy as np
# import pandas as pd
# from glob import glob
import tensorflow as tf

# Data
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
# from tensorflow.keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 

# from tensorflow.keras import layers
# from tensorflow.keras import models

# Callbacks 
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tf_explain.callbacks.grad_cam import GradCAM
#https://tf-explain.readthedocs.io/en/latest/usage.html

# Metrics
from tensorflow.keras.metrics import MeanIoU

# Based on Yoonjung's Jupyter notebook helper functions for attention unet
# James integrated her code into our BC segmentation system app
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