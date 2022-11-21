import torch
from torchvision import transforms


IMG_SIZE = (255, 255)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_PATH = r'../../Models/pretrained_model.h5'


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform(image)


def classify(image):
    # apply preprocessing
    image = preprocess(image)
    # load the saved model
    model = torch.load(MODEL_PATH)
    with torch.no_grad():
        pred = model(image)
        pred = torch.exp(pred)

    return pred
