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

class ActionRunLesionClassification(Action):
    def name(self):
        return "action_run_lesion_classification"
    
    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Running Breast Cancer Classifier")

class ActionRunLesionSegmentation()
    def name(self):
        return "action_run_lesion_segmentation"
    
    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Running Breast Cancer Segmentation")

class ActionRunLesionVolRendering()
    def name(self):
        return "action_run_lesion_vol_rendering"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Running Breast Cancer Volume Rendering")