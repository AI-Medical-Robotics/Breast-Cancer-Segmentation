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
# quantitative

def evaluate_attention_unet(trained_unet):
    train_loss, train_acc, train_iou, val_loss, val_acc, val_iou = trained_unet.history.values()

    plt.figure(figsize=(20,8))

    plt.subplot(1,3,1)
    plt.title("Model Loss")
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Validtion")
    plt.legend()
    plt.grid()

    plt.subplot(1,3,2)
    plt.title("Model Accuracy")
    plt.plot(train_acc, label="Training")
    plt.plot(val_acc, label="Validtion")
    plt.legend()
    plt.grid()

    plt.subplot(1,3,3)
    plt.title("Model IoU")
    plt.plot(train_iou, label="Training")
    plt.plot(val_iou, label="Validtion")
    plt.legend()
    plt.grid()

    plt.show()



# qualitative