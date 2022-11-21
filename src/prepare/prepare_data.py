# Prepare data for training
from utils.utils import loadImages, SIZE

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

# Based on Yoonjung's Jupyter notebook attention unet data prep code
# James integrated her code into our BC segmentation system app
def prepare_busi_data(prep_train = True):
    # TODO: Pass an option to prepare data for training/validation or testing...
        # Already accounted for training, need to account for deployment
    # TODO: update base path "/media/james/My Passport/", make it more adaptable
    file_path = '/media/james/My Passport/Jetson_TX2_CMPE258/Dataset_BUSI_with_GT/'
    labels = sorted(os.listdir(file_path))
    labels

    ori_mask_paths = sorted([sorted(glob(file_path + name + "/*mask.png")) for name in labels])
    # print("ori_mask_paths = ", ori_mask_paths)
    print(ori_mask_paths[0][0])
    print(ori_mask_paths[1][0])
    print(ori_mask_paths[2][0])

    image_paths = []
    mask_paths = []

    for label in ori_mask_paths:
        for path in label:
            img_path = path.replace('_mask','')
            #add original images
            image_paths.append(img_path)
            #add mask images 
            mask_paths.append(path)

    num_train_set = len(image_paths) * 0.8
    # num_test_set = len(image_paths) - num_train_set
    if prep_train:
        # get the first 80% of image paths for training set
        image_paths = image_paths[:num_train_set]
    else:
        # get the last 20% of image paths for testing set,
        # everything remaining after training set
        image_paths = image_paths[num_train_set:]

    print('len(image_paths)', len(image_paths))
    print('len(mask_paths)', len(mask_paths))

    # print('image_paths ', image_paths[0])
    # showImage(loadOneImage(image_paths[0], SIZE))

    # print('image_paths ', image_paths[0])
    # print('mask_paths ', image_paths[0])
    # showMask(loadOneImage(image_paths[0], SIZE), loadOneImage(mask_paths[0], SIZE)[:,:,0], alpha=0.6)

    # print('image_paths ', image_paths[500])
    # print('mask_paths ', mask_paths[500])
    # showMask(loadOneImage(image_paths[500], SIZE), loadOneImage(mask_paths[500], SIZE)[:,:,0], alpha=0.6)

    # img_bg = np.zeros((1,SIZE,SIZE,3))
    # mask1 = loadOneImage('/content/drive/MyDrive/DL/archive/Dataset_BUSI_with_GT/benign/benign (4)_mask.png', SIZE)
    # mask2 = loadOneImage('/content/drive/MyDrive/DL/archive/Dataset_BUSI_with_GT/benign/benign (4)_mask_1.png', SIZE)

    # combined_mask = img_bg + mask1 + mask2
    # combined_mask = combined_mask[0,:,:,0]
    # showImage(combined_mask, cmap='gray')

    # showImage(loadOneImage('/content/drive/MyDrive/DL/archive/Dataset_BUSI_with_GT/benign/benign (4).png', SIZE))
    # plt.imshow(combined_mask, cmap='gray', alpha=0.5)
    # plt.axis('off')
    # plt.show()

    images = loadImages(image_paths, SIZE)
    masks = loadImages(mask_paths, SIZE, mask=True)

    # print("images.shape = {}".format(images.shape))
    # print("masks.shape = {}".format(masks.shape))

    # plt.figure(figsize=(13,8))
    # for i in range(9):
    #     ax = plt.subplot(3,3,i+1)
    #     id = np.random.randint(len(images))
    #     ax.set_title(id)
    #     showMask(images[id], masks[id], cmap='terrain')
    # plt.show()

    # plt.figure(figsize=(13,8))
    # for i in range(9):
    #     ax = plt.subplot(3,3,i+1)
    #     id = np.random.randint(len(images))
    #     ax.set_title(id)
    #     showMask(images[id], masks[id], cmap='binary')
    # plt.show()

    # plt.figure(figsize=(13,8))
    # for i in range(9):
    #     ax = plt.subplot(3,3,i+1)
    #     id = np.random.randint(len(images))
    #     ax.set_title(id)
    #     showMask(images[id], masks[id], cmap='afmhot')
    # plt.show()
    return images, masks