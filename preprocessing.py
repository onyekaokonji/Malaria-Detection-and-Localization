# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

def load_and_preprocess_images(base_path):
    Image_files = []
    Labels = []
    Dataset = []
  
    for label in os.listdir(base_path):

    # append labels from sub-directory names
        Labels.append(label)
    
    # creating image path
        images_path = os.path.join(base_path, label)
    
        for image_name in os.listdir(images_path):

            image_paths = os.path.join(images_path, image_name)
      
      # read the images
            img = cv2.imread(image_paths)

            try:
        
        # resizing the images
                img = cv2.resize(img, (64, 64), cv2.INTER_CUBIC)

        # make copy of resized image for edge detection
                img_copy = img.copy()
    
        # convert the images to grayscale for edge detection
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)   
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

        # preprocessing using canny edge detector to make image edges sharper and image smoother
                img_copy = cv2.GaussianBlur(img, (7,7), 0) 
                edges = cv2.Canny(img_copy, threshold1 = 80, threshold2 = 160)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # superimposing image with detected edges
                final_img = cv2.addWeighted(img, 0.5, edges, 0.5, 0)
        
                Dataset.append([final_img, np.array(label)])
            except:
                continue

    return Dataset