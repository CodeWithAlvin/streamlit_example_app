import cv2
import numpy as np
import tensorflow as tf

def load_model(path):
    # loading the model
    model = tf.keras.models.load_model(path, compile=False)
    return model


def predict(image,model):
    # model allowable image dimensions
    img_width, img_height = 224, 224
    # loading the image
    image = cv2.resize(image, (img_width, img_height))
    # coverting BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # adding a new axis [batch axis]
    image = image[np.newaxis, ...]
    # rescale image from [0,255] to [0,1]
    image = image / 255.
    # predicting the output
    prediction = model.predict(image)
    # squeeze/remove an extra axis
    prediction = np.squeeze(prediction)

    return prediction[:3]