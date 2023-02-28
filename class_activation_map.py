import os
import cv2
import numpy as np
from keract import display_activations, get_activations, display_heatmaps
from tensorflow.keras.models import load_model

def class_activation_map(img_path:str, model_path:str):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (75, 75))
  img = np.expand_dims(img, 0)
  img = img/255.
  act_model = load_model(model_path)
  activations = get_activations(act_model, img)
  # print(display_activations(activations))
  display_heatmaps(activations, img, save=False)


if __name__ == '__main__':
  class_activation_map()