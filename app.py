import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C://Users//vamsi//Desktop//mahesh//plant_disease_model.h5')

# Define the classes
classes = ['Pepper_bell__Bacterial_spot',
           'Pepper_bell__healthy',
           'Potato___Early_blight',
           'Potato___Late_blight',
           'Potato___healthy',
           'Tomato_Bacterial_spot',
           'Tomato_Early_blight',
           'Tomato_Late_blight',
           'Tomato_Leaf_Mold',
           'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite',
           'Tomato__Target_Spot',
           'Tomato_Tomato_YellowLeaf_Curl_Virus',
           'Tomato__Tomato_mosaic_virus',
           'Tomato_healthy']

# Streamlit App
st.title("Plant Disease Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (224, 224))

    st.image(image, caption="Uploaded Image.", use_column_width=True)

    image = image / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_class = classes[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"The model predicts: {predicted_class}")
    st.subheader("Class Probabilities:")
    for i in range(len(classes)):
        st.write(f"{classes[i]}: {prediction[i]:.4f}")


   

# To run the app, use the following command in your terminal or command prompt:
# streamlit run app.py