# quantitative

def evaluate_attention_unet():
    train_loss, train_acc, train_iou, val_loss, val_acc, val_iou = results_epo25.history.values()

    plt.figure(figsize=(20,8))

    plt.subplot(1,3,1)
    plt.title("Model Loss")
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Validtion")
    plt.legend()
    plt.grid()

    plt.subplot(1,3,2)
    plt.title("Model Accuracy")
    plt.plot(train_acc, label="Training")
    plt.plot(val_acc, label="Validtion")
    plt.legend()
    plt.grid()

    plt.subplot(1,3,3)
    plt.title("Model IoU")
    plt.plot(train_iou, label="Training")
    plt.plot(val_iou, label="Validtion")
    plt.legend()
    plt.grid()

    plt.show()

# qualitative