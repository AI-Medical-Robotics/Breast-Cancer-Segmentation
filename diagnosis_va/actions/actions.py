# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import os
import cv2
import time
import random
import numpy as np
import pandas as pd

# from typing import Any, Text, Dict, List
from rasa_sdk import Action #, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version

import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

from prepare.prepare_data import prepare_busi_data
from train.train_bcs_models import train_attention_unet
from evaluate.evaluate_bcs_models import evaluate_attention_unet
from deploy.deploy_bcs_models import deploy_model


class ActionTrainLesionSegmentation(Action):
    def name(self):
        return "action_train_lesion_segmentation"
    
    def run(self, dispatcher, tracker, domain):
        model_name = tracker.get_slot("model_seg_name")

        if model_name == "Attention UNet" or model_name == "attention unet":
            bc_ultrasound_data_path = '/media/james/My Passport/Jetson_TX2_CMPE258/Dataset_BUSI_with_GT/'
            images, masks = prepare_busi_data(bc_ultrasound_data_path, prep_train=True)
            trained_att_unet = train_attention_unet(images, masks)
            evaluate_attention_unet(trained_att_unet)
        else:
            print("Attention UNet only breast cancer segmentation supported model")
        dispatcher.utter_message(text="Running Breast Cancer Segmentation Training")
        return [SlotSet("model_seg_name", None)]

class ActionRunLesionClassification(Action):
    def name(self):
        return "action_run_lesion_classification"
    
    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Running Breast Cancer Classifier")

class ActionRunLesionSegmentation(Action):
    def name(self):
        return "action_run_lesion_segmentation"
    
    def run(self, dispatcher, tracker, domain):

        model_name = tracker.get_slot("model_seg_name_deploy")

        if model_name == "Attention UNet" or model_name == "attention_unet":
            attent_unet_path = "/home/james/proj/james/Breast-Cancer-Segmentation/diagnosis_va/actions/AttentioUnetModel.h5"
            # attent_unet_path = "/home/james/proj/james/Breast-Cancer-Segmentation/src/AttentioUnetModel.h5"

            bc_ultrasound_data_path = '/media/james/My Passport/Jetson_TX2_CMPE258/Dataset_BUSI_with_GT/'
            test_images, test_masks = prepare_busi_data(bc_ultrasound_data_path, prep_train=False)
            attention_unet = deploy_model(attent_unet_path, bc_ultrasound_data_path, test_images, test_masks, show_results=True)
            save_pred_bc_seg_file = attention_unet.display_prediction()
        else:
            print("Attention UNet only supported model for deployment")

        dispatcher.utter_message(image=save_pred_bc_seg_file)
        # reset slot
        return [SlotSet("model_seg_name_deploy", None)]

class ActionRunLesionVolRendering(Action):
    def name(self):
        return "action_run_lesion_vol_rendering"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Running Breast Cancer Volume Rendering")