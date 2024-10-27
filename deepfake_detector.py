import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model_path = r'C:\\Users\\asus\\OneDrive\\Desktop\\ML(DEEPFAKE)\\deepfake_detector_model_improved_with_weights.h5'
model = load_model(model_path)


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0
    return img_array

st.title("Deepfake Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = image.load_img(uploaded_file)
    img_array = preprocess_image(img)

    
    prediction = model.predict(img_array)
    

    prediction_label = 'Fake' if prediction[0] < 0.5 else 'Real'


    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Prediction: **{prediction_label}**")
