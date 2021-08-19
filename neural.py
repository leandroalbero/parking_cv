import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def hello():
    print("Number of gpus available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    inputs = keras.Input(shape=(150,150,3))
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs= outputs, name="mnist_model")
    model.summary()

    pass
