# train attention UNet
from bcs_models import attention_unet
from utils import showMask, showImage

import argparse
import datetime
import re


# class train_bcs_models:
#     pass

class report(Callback):
    def __init__(self, images, masks):
        self.m_images = images
        self.m_masks = masks
        
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

def train_attention_unet(images, masks):
    print("images.shape[-3:] = {}".format(images.shape[-3:]))
    print("images.shape = {}".format(images.shape))
    print("masks.shape = {}".format(masks.shape))

    model = attention_unet(images, masks)

    cb = [
        ModelCheckpoint("AttentioUnetModel.h5", save_best_only=True),
        report()
    ]

    BATCH_SIZE = 10
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
    # return trained attention UNet
    return results_epo25

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--local_rank', type=int, default=None)
parser.add_argument('--cfg', default='res101_coco', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--img_size', default=544, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=4000, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

