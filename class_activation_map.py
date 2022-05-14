import os
from keract import display_activations, get_activations, display_heatmaps
from tensorflow.keras.models import load_model

def class_activation_map(img_path, model_path):
    img = img_path.reshape(1,64,64,3)
    act_model = load_model(model_path)
    activations = get_activations(act_model, img)
    print(display_activations(activations))
    display_heatmaps(activations, img, save=False)