# train attention UNet
from prepare.prepare_data import prepare_busi_data
from train.train_bcs_models import train_attention_unet

# TODO: As we get more models to train, we'll add conditions
# Run the data prep function, pass images, masks to train attention UNet
images, masks = prepare_busi_data()
train_attention_unet(images, masks)
