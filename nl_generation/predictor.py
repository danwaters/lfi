import os
import pickle

import numpy as np
import tensorflow as tf

WORD_COUNT = 100
WINDOW_SIZE = 6 # How many words (max) to feed back to the model

class MyPredictor(object):
  def __init__(self, model, preprocessor):
    self._model = model
    self._preprocessor = preprocessor
    
    self._outputs = []
    self._window = []

  def predict(self, instances, **kwargs):
    inputs = np.asarray(instances)
    words = 0

    while words < WORD_COUNT:
      outputs = self._model.predict(inputs)
      word_index = np.argmax(outputs, axis=-1)

      # Add the prediction to the list of outputs and the sliding window
      self._outputs.append(word_index)
      self._window.append(word_index)

      # maintain the sliding window
      if len(self._window) > WINDOW_SIZE:
        self._window = self._window[:WINDOW_SIZE]

      words += 1

    return self._outputs

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'saved_model.pb')
    model = tf.keras.models.load_model(model_path)

    preprocessor_path = os.path.join(model_dir, './preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
      preprocessor = pickle.load(f)

    return cls(model, preprocessor)