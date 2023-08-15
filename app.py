import streamlit as st
import numpy as np
import PIL.Image
import pandas as pd
import os
from fastai.vision.all import Path,load_learner,Image
import pathlib
import platform

plt = platform.system()

if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

learn =  load_learner(pathlib.Path()/'train/r0858516_Birds.pkl')

df = pd.read_csv(pathlib.Path()/"train/labels.csv",index_col=['label'])
#/mount/src/birds/requirements.txt    

def get_name(name):
    return df[df.index == name].reset_index(drop=True)['Bird'][0]

def predict_img(img, learn):
    #pil_img = PIL.Image.open(img)
    img = np.asarray(img) # Image to display   
    return learn.predict(
    item=img
)

 
html_temp = """
    <div style="background-color:#f63366;padding:10px;margin-bottom: 25px">
    <h2 style="color:white;text-align:center;">Car Model Classification App</h2>
    <p style="color:white;text-align:center;" >This is a <b>Streamlit</b> app use for prediction of the <b>7 types of Birds</b>.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

#image = PIL.Image.open('bg.jpg')
#st.image(image, use_column_width=True)

option = st.radio('', ['Choose a test image', 'Choose your own image'])
if option == 'Choose your own image':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg") #file upload
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        pred_class, prob , test = predict_img(img, learn)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width=200)
        with col2:
            st.success("Car Name:  [" + str(pred_class) + "] ")
            st.info("Probability: [" + str(prob) + '%]')
        
else:
    test_images = os.listdir('Sample_Images')
    test_image = st.selectbox('Please select a test image:', test_images)
    file_path = 'Sample_Images/' + test_image
    img = PIL.Image.open(file_path)
    pred_class, prob , test = predict_img(img, learn)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, width=200)
    with col2:
        st.success("Car Name Name:  [" + str(pred_class) + "] ")
        st.info("Probability: [" + str(prob) + '%]')
    
