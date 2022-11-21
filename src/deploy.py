
from prepare.prepare_data import prepare_busi_data
from deploy.deploy_bcs_models import deploy_model

attent_unet_path = "/home/james/proj/james/Breast-Cancer-Segmentation/src/AttentioUnetModel.h5"

# TODO: In prepare_data.py, Pass an option to prepare data for training/validation or testing...
    # Already accounted for training, need to account for deployment
    # that could be a way of getting these test images and masks.
    # Ex: "test_images, test_masks = prepare_busi_data(test=True)". Maybe theres a preset 80% training & 20% test
test_images, test_masks = prepare_busi_data(train=False)
# test_images = None
# test_masks = None

attention_unet = deploy_model(attent_unet_path, test_images, test_masks, show_results=True)

attention_unet.display_prediction()