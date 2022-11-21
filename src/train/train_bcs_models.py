# train attention UNet
from prepare.prepare_data import prepare_busi_data
from models.bcs_models import attention_unet
from utils.utils import showMask, showImage

# import time
# import argparse
# import datetime
# import re

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

# class train_bcs_models:
#     pass

class report(Callback):
    def __init__(self, images, masks):
        self.m_images = images
        self.m_masks = masks
        
    def on_epoch_end(self, epochs, logs=None):
        id = np.random.randint(200)
        explainer = GradCAM()
        image = self.m_images[id]
        mask = self.m_masks[id]
        pred_mask = self.model.predict(image[np.newaxis,...])
        cam_explain = explainer.explain(
            validation_data=(image[np.newaxis,...], mask),
            class_index=1,
            layer_name='Attention4',
            model=self.model
        )

        #show results:
        plt.figure(figsize=(10,5))

        plt.subplot(1,3,1)
        plt.title("Original Image With Mask")
        showMask(image, mask, cmap='afmhot')

        plt.subplot(1,3,2)
        plt.title("Original Image With Predicted Mask")
        showMask(image, pred_mask, cmap='afmhot')

        plt.subplot(1,3,3)
        showImage(cam_explain, title="GradCAMCallback")

        plt.tight_layout()
        plt.show()

def train_attention_unet(images, masks):
    print("images.shape[-3:] = {}".format(images.shape[-3:]))
    print("images.shape = {}".format(images.shape))
    print("masks.shape = {}".format(masks.shape))

    model = attention_unet(images, masks)

    # Save the model.
    cb = [
        ModelCheckpoint("AttentioUnetModel.h5", save_best_only=True),
        report()
    ]

    BATCH_SIZE = 10
    SPE = len(images)//BATCH_SIZE
    print(SPE)

    # Training
    results_epo25 = model.fit(
        images, masks,
        validation_split=0.2,
        epochs=25,
        steps_per_epoch=SPE,
        batch_size=BATCH_SIZE,
        callbacks=cb
    )

    # return trained attention UNet or saved file location .h5
    return results_epo25

# parser = argparse.ArgumentParser(description='Breast Cancer Seg Training Script')
# parser.add_argument('--local_rank', type=int, default=None)
# parser.add_argument('--cfg', default='res101_coco', help='The configuration name to use.')
# parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
# parser.add_argument('--img_size', default=544, type=int, help='The image size for training.')
# parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
# parser.add_argument('--val_interval', default=4000, type=int,
#                     help='The validation interval during training, pass -1 to disable.')
# parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
# parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
# parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

# args = parser.parse_args()
# cfg = get_config(args, mode='train')
# cfg_name = cfg.__class__.__name__

# TODO: As we get more models to train, we'll add conditions
# Run the data prep function, pass images, masks to train attention UNet
images, masks = prepare_busi_data()
train_attention_unet(images, masks)


