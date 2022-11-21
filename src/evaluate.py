from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version
from evaluate.evaluate_bcs_models import evaluate_attention_unet

"""
File "/home/james/proj/james/Breast-Cancer-Segmentation/src/evaluate/evaluate_bcs_models.py", line 36, in evaluate_attention_unet
    train_loss, train_acc, train_iou, val_loss, val_acc, val_iou = trained_unet.history.values()
AttributeError: 'NoneType' object has no attribute 'values'
"""
attent_unet = load_model("/home/james/proj/james/Breast-Cancer-Segmentation/src/AttentioUnetModel.h5")
evaluate_attention_unet(attent_unet)