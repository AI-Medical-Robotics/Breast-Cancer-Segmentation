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


def attention_unet(): 
    inputs = tf.keras.Input(shape=(256, 256, 3))
    n_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    C1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(n_inputs)
    D1 = tf.keras.layers.Dropout(0.1)(C1)
    C12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D1)
    M1 = tf.keras.layers.MaxPooling2D((2, 2))(C12)

    C2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(M1)
    D2 = tf.keras.layers.Dropout(0.1)(C2)
    C22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D2)
    M2 = tf.keras.layers.MaxPooling2D((2, 2))(C22)
    
    C3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(M2)
    D3 = tf.keras.layers.Dropout(0.2)(C3)
    C32 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D3)
    M3 = tf.keras.layers.MaxPooling2D((2, 2))(C32)
    
    C4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(M3)
    D4 = tf.keras.layers.Dropout(0.2)(C4)
    C42 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D4)
    M4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(C42)
    
    C5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(M4)
    D5 = tf.keras.layers.Dropout(0.3)(C5)
    C52 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D5)

    #Expansive path 
    T6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(C52)
    CO6 = tf.keras.layers.concatenate([T6, C42])
    C6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(CO6)
    D6 = tf.keras.layers.Dropout(0.2)(C6)
    C62 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D6)
    
    T7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(C62)
    CO7 = tf.keras.layers.concatenate([T7, C32])
    C7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(CO7)
    D7 = tf.keras.layers.Dropout(0.2)(C7)
    C72 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D7)
    
    T8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(C72)
    CO8 = tf.keras.layers.concatenate([T8, C22])
    C8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(CO8)
    D8 = tf.keras.layers.Dropout(0.1)(C8)
    C82 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C8)
    
    T9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(C82)
    CO9 = tf.keras.layers.concatenate([T9, C12], axis=3)
    C9 = tf.keras.layers.Conv2D(32, (3, 3),name="Attention4", activation='relu', kernel_initializer='he_normal', padding='same')(CO9)
    D9 = tf.keras.layers.Dropout(0.1)(C9)
    C92 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(D9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1),padding='same', activation='sigmoid')(C92)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2, name='IoU')])

    model.summary()
    return model
