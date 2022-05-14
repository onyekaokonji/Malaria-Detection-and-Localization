import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
#import cv2
from tensorflow.keras.models import load_model
import zipfile
import tempfile

st.set_page_config(page_title='Malaria Detector', layout = 'centered', initial_sidebar_state='auto')

st.title('Group LSTM Malaria Detector')

#st.write("""
          # Malaria Detection
          #"""
          #)
          
upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")

model_upload = st.sidebar.file_uploader("Upload Saved Model", type = "zip")

if model_upload is not None:
  myzipfile = zipfile.ZipFile(model_upload)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = tf.keras.models.load_model(model_dir)

Generate_pred=st.sidebar.button("Predict")

def load_and_preprocess_test_images(img):

  img = img.resize((64,64))
  img_copy = img.copy()
  img_copy = img_copy.convert('L')
  #img_copy = img_copy.filter(ImageFilter.GaussianBlur(radius = 5))
  edges = img_copy.filter(ImageFilter.FIND_EDGES)
  edges = edges.convert('RGB')
  #edges = ImageOps.fit(edges, img_copy.size, Image.ANTIALIAS)
  #img_copy = img_copy.convert('RGB')
 # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
  final_img = Image.blend(img, edges, 0.5)
  final_img = np.expand_dims(final_img, axis = 0)
  final_img = final_img/255.

  return final_img
   
def import_n_pred(image_data, model):
    
    size = (64,64)
    img_data = image_data.resize(size)
    # img = np.asarray(image)
    img = load_and_preprocess_test_images(img_data)
    pred = model.predict(img)

    return pred
    
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    class_indices = np.argmax(pred, axis = 1)
    if class_indices == 0:
        class_label = 'Uninfected'

    else:
        class_label = 'Parasitized'
 
    st.title("Blood sample is {}".format(class_label))
    
    



