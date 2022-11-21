# train attention UNet
# import time
# import argparse
# import datetime
# import re

from prepare.prepare_data import prepare_busi_data
from train.train_bcs_models import train_attention_unet
from evaluate.evaluate_bcs_models import evaluate_attention_unet

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

# Based on Yoonjung's Jupyter notebook for training unet
# James added functions to make training easier to follow and run from script

# TODO: As we get more models to train, we'll add conditions
# Run the data prep function, pass images, masks to train attention UNet
# default split 80% for training set
bc_ultrasound_data_path = '/media/james/My Passport/Jetson_TX2_CMPE258/Dataset_BUSI_with_GT/'
images, masks = prepare_busi_data(bc_ultrasound_data_path, train=True)
trained_att_unet = train_attention_unet(images, masks)
evaluate_attention_unet(trained_att_unet)