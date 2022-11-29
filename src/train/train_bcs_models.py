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

# Based on Yoonjung's Jupyter notebook train attention unet code
# James integrated her code into our BC segmentation system app
class report(Callback):
    def __init__(self, images, masks, show_results=False):
        self.m_images = images
        self.m_masks = masks
        self.m_show_results = show_results
        self.m_us_train_plots_path = "train_plots/"
        
    def on_epoch_end(self, epochs, logs=None):
        id = np.random.randint(len(self.m_images))
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

        f = plt.figure(figsize=(10,5))

        plt.subplot(1,3,1)
        plt.title("Original Image With Mask")
        showMask(image, mask, cmap='afmhot')

        plt.subplot(1,3,2)
        plt.title("Original Image With Predicted Mask")
        showMask(image, pred_mask, cmap='afmhot')

        plt.subplot(1,3,3)
        showImage(cam_explain, title="GradCAMCallback")

        plt.tight_layout()

        if self.m_show_results:
            plt.show()

        # TODO: save figs, turn into video

        plt.close(f)
        # add else block to save results as jpgs
        if epochs % 5 == 0:
            print("Training AttentionUNet: Current Epochs = {}".format(epochs))

def train_attention_unet(images, masks):
    print("images.shape[-3:] = {}".format(images.shape[-3:]))
    print("images.shape = {}".format(images.shape))
    print("masks.shape = {}".format(masks.shape))

    model = attention_unet()

    # Save the model.
    cb = [
        ModelCheckpoint("AttentionUnetModel.h5", save_best_only=True),
        report(images, masks)
    ]

    BATCH_SIZE = 4
    SPE = len(images)//BATCH_SIZE
    print(SPE)

    # Training
    # switching from 25 epochs to 20 epochs, TF warned input ran out of data
    # I noticed this problem for 20 epochs too.
    # TF: WARNING:tensorflow:Your input ran out of data; interrupting training. 
    # Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` 
    # batches (in this case, 3900 batches)
    results_epo25 = model.fit(
        images, masks,
        validation_split=0.2,
        epochs=25,
        callbacks=cb
    )

    # return trained attention UNet or saved file location .h5
    return results_epo25
