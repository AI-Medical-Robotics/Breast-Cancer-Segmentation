# import time
# import argparse
# import datetime
# import re

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


from models.bcs_models import attention_unet
from utils.utils import showMask, showImage

# class train_bcs_models:
#     pass

class report(Callback):
    def __init__(self, images, masks, show_results):
        self.m_images = images
        self.m_masks = masks
        self.m_show_results = show_results
        
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
        if self.m_show_results:
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
        # add else block to save results as jpgs

def train_attention_unet(images, masks):
    print("images.shape[-3:] = {}".format(images.shape[-3:]))
    print("images.shape = {}".format(images.shape))
    print("masks.shape = {}".format(masks.shape))

    model = attention_unet()

    # Save the model.
    cb = [
        ModelCheckpoint("AttentioUnetModel.h5", save_best_only=True),
        report(images, masks)
    ]

    BATCH_SIZE = 4
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
