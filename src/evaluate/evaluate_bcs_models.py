# common
import os
import uuid
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
def evaluate_attention_unet(trained_unet, trained_results_path):
    if not os.path.exists(trained_results_path):
        os.makedirs(trained_results_path)

    # must pass model after its been trained for getting history
    train_loss, train_acc, train_iou, val_loss, val_acc, val_iou = trained_unet.history.values()

    f = plt.figure(figsize=(20,8))

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

    # plt.show()

    save_plot_metrics_file = "{}/train_val_loss_acc_iou_{}.jpg".format(trained_results_path, uuid.uuid4())
    f.savefig(save_plot_metrics_file, bbox_inches="tight")

    plt.close(f)
    return save_plot_metrics_file