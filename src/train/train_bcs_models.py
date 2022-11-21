# train attention UNet
from bcs_models import attention_unet

# class train_bcs_models:
#     pass

class report(Callback):
    def on_epoch_end(self, epochs, logs=None):
        id = np.random.randint(200)
        explainer = GradCAM()
        image = images[id]
        mask = masks[id]
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

def train_attention_unet():
    model = attention_unet()

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