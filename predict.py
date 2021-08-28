import os
import tensorflow.keras.models
from PIL import Image
import numpy as np


def predict_image(path):
    model = tensorflow.keras.models.load_model('model3.h5')
    files = os.listdir(path)
    predictions = {}
    for file in files:
        im = Image.open(f"{path}{file}")
        im = im.resize((150, 150))
        im = np.expand_dims(im, axis=0)
        im = np.array(im)
        im = im / 255
        if model.predict(im)[0][0] > 0.5:
            predictions[file] = 1
        else:
            predictions[file] = 0
    return predictions