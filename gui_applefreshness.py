#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
model = tf.keras.models.load_model('inceptionv3_model.hdf5')


# In[2]:


import streamlit as st
st.write("""
         # Apple Freshness Grading
         """
         )
st.write("This system is able to classify apple into 3 categories : fresh, medium fresh, rotten.")
st.write("It may not be 100% accurate but it is still reliable.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


# In[3]:


import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a fresh apple!")
    elif np.argmax(prediction) == 1:
        st.write("It is a medium fresh apple!")
    else:
        st.write("It is a rotten apple!")
    
    st.text("Probability (0: freshapples, 1: mediumfresh, 2: rottenapples")
    st.write(prediction)


# In[ ]:




