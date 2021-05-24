
from net import SimpleNet
from main import getImagePath

import cv2
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image

# PATH = "models/faceModel_23.model"  # Train acc: 93.33%, Test acc: 76.36%
PATH = "models/faceModel_31_good.model"  # Train acc: 98.66%, Test acc: 75.15%
# PATH = "models/faceModel_29_good.model"
CONFIDENCE_THRESHOLD = 0.85

def inference(model, image):
    model.eval()

    return predictImage(model, image)


def predictImage(model, image):
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)
    m = nn.Softmax(dim=1)
    output = m(output)
    # index = output.data.numpy().argmax()
    confidence = output.data.numpy()[0][0]
    # print(output)

    # return index == 0, confidence
    return confidence

if __name__ == "__main__":

    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (15, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    # load the model
    model = SimpleNet(2)
    model.load_state_dict(torch.load(PATH))

    video_capture = cv2.VideoCapture(0)  # internal web-cam
    while True:
        # (, last frame)
        _, frame = video_capture.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        confidence = inference(model, img)

        text = "Face detected: False {}%".format(round(confidence * 100, 2))
        if confidence >= CONFIDENCE_THRESHOLD:
            text = "Face detected: True {}%".format(round(confidence * 100, 2))

        cv2.putText(frame, text, topLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
