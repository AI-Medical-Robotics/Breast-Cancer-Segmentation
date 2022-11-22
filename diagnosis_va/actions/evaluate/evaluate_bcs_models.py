# common
import os
import tensorflow.keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf

# Data
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 

from tensorflow.keras import layers
from tensorflow.keras import models

# Callbacks 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tf_explain.callbacks.grad_cam import GradCAM
#https://tf-explain.readthedocs.io/en/latest/usage.html

# Metrics
from tensorflow.keras.metrics import MeanIoU

# Based on Yoonjung's Jupyter notebook evaluate attention unet code
# James integrated her code into our BC segmentation system app
def evaluate_attention_unet(trained_unet, show_results=False):
    if show_results:
        # TODO: Debug history.values() since I get no values. Probably need the UNet to be fully trained
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
    else:
        print("Saving training metrics not supported yet")