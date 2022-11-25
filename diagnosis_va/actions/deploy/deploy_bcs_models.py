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


from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version


from models.bcs_models import attention_unet
from utils.utils import showMask, showImage

# class deploy_bcs_models:
#     pass

# Based on Yoonjung's Jupyter notebook train attention unet code
# James integrated her code into our BC segmentation system app
class deploy_model:
    def __init__(self, model_path, test_images, test_masks, show_results=False):
        self.m_test_images = test_images
        self.m_test_masks = test_masks
        self.m_show_results = show_results
        self.m_model = load_model(model_path)
        
    def display_prediction(self):
        id = np.random.randint(200)
        explainer = GradCAM()
        test_image = self.m_test_images[id]
        test_mask = self.m_test_masks[id]
        pred_mask = self.m_model.predict(test_image[np.newaxis,...])
        cam_explain = explainer.explain(
            validation_data=(test_image[np.newaxis,...], test_mask),
            class_index=1,
            layer_name='Attention4',
            model=self.m_model
        )

        #show results:
        if self.m_show_results:
            plt.figure(figsize=(10,5))

            plt.subplot(1,3,1)
            plt.title("Original Test Image With Test Mask")
            showMask(test_image, test_mask, cmap='afmhot')

            plt.subplot(1,3,2)
            plt.title("Original Test Image With Predicted Mask")
            showMask(test_image, pred_mask, cmap='afmhot')

            plt.subplot(1,3,3)
            showImage(cam_explain, title="GradCAMCallback")

            plt.tight_layout()
            plt.show()
        # add else block to save results as jpgs

