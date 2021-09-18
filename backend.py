import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import CustomObjectScope

def load_model(path,custom_objects=None):
    """
    path ==> path to the model
    custom_objects ==> dictonary of the custom objects
      example : {"KerasLayer":KerasLayer, "FixedDropout": FixedDropout }
      first will be name used when model is trained and second will be name in current file
    """
    # loading the model
    # if model uses custom objects they will be configured here
    if custom_objects is not None:
        with CustomObjectScope(custom_objects):
            model = tf.keras.models.load_model(path,)
        return model
    # else load model directly
    model = tf.keras.models.load_model(path, compile=False)
    return model


def predict(image,model):
    # labels during training
    labels = {0: 'NORMAL', 1: 'PNEUMONIA', 2: 'TUBERCULOSIS'}
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
    # getting the index with maximum probability
    max_probability = np.argmax(prediction)
    # getting label name 
    result = labels.get(max_probability)
    # creating dataframe of prediction result
    raw_data = []
    for index,name in labels.items():
        # appending name , coresponding prediction value 
        # example [pneumonia, 0.5]
        raw_data.append([name, prediction[index]])
    # making dataframe of it
    dataframe=pd.DataFrame(raw_data,columns=["disease","accuracy"])
    
    return result,dataframe