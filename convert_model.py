import tensorflow as tf
from tensorflow.python.keras import models


model = models.load_model('models/actor.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/actor.tflite', 'wb') as fp:
    fp.write(tflite_model)
