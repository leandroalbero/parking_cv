import os
import tensorflow.keras.models
from PIL import Image
import numpy as np


def predict_patch(path, model=None):
    if model is None:
        model = tensorflow.keras.models.load_model('model.h5')
    im = Image.open(path)
    im = im.resize((150, 150))
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    im = im / 255
    return model.predict(im)


def predict_image(path, model=None):
    if model is None:
        model = tensorflow.keras.models.load_model('model.h5')
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
