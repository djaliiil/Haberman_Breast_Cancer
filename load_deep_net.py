import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib

def main(input):
    scaler = joblib.load('data/scaler.save')
    input = pd.DataFrame(input)
    input_scaled = scaler.transform(input)
    model = tf.keras.models.load_model('data/model_file.h5')
    predict = model.predict(input_scaled)
    val = np.argmax(predict)
    return val
