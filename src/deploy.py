
from prepare.prepare_data import prepare_busi_data
from deploy.deploy_bcs_models import deploy_model

attent_unet_path = "/home/james/proj/james/Breast-Cancer-Segmentation/src/AttentioUnetModel.h5"

# Prep data: either 80% training or 20% test
bc_ultrasound_data_path = '/media/james/My Passport/Jetson_TX2_CMPE258/Dataset_BUSI_with_GT/'
test_images, test_masks = prepare_busi_data(bc_ultrasound_data_path, prep_train=False)

attention_unet = deploy_model(attent_unet_path, bc_ultrasound_data_path, test_images, test_masks, show_results=True)

save_pred_bc_seg_file = attention_unet.display_prediction()
# dispatcher.utter_message(image=save_pred_bc_seg_file)