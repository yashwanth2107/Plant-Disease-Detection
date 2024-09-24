import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from io import BytesIO
import cv2

# Custom CSS styles
custom_css = """
<style>
    body {
        background-color: #f4f4f4; /* Background color of the entire page */
        color: #333; /* Text color */
        font-family: Arial, sans-serif; /* Font family */
    }

    .stApp {
        max-width: 1200px; /* Set the maximum width of the app */
        margin: 0 auto; /* Center the app on the page */
    }

    .custom-radio input:checked + span {
        background-color: #4CAF50;
        color: white;
    }

    .custom-radio input + span {
        padding: 8px 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: background-color 0.3s;
    }

    .custom-radio input {
        position: absolute;
        opacity: 0;
    }

    .selected-tab {
        background-color: lightgreen;
        color: green;
        padding: 8px 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
        display: inline-block;
        margin-right: 8px;
    }

    .selected-tab:hover {
        background-color: limegreen; /* Change color on hover */
    }

    .sidebar .stRadio span {
        font-size: 16px; /* Set font size for radio buttons in the sidebar */
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Load the model
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']  # Replace with your class labels

model = tf.keras.models.load_model('C://Users//banal//OneDrive//Desktop//pdd//difclasses.h5')

# Function to identify crop disease
def identify_crop_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    disease_name = class_labels[predicted_class_index]

    return disease_name

# Main function
def main():
    st.title("Plant Disease Detection")
    st.subheader("An AI-powered solution for farmers")

    # Sidebar with options
    st.sidebar.title("Options")
    selected_option = st.sidebar.radio("Choose an option", ("Upload an image", "Capture an image"))

    # Content area
    st.write("Welcome to our Plant Disease Detection and Medication Recommendation tool!")
    st.markdown("---")

    # Highlight the selected tab
    # st.markdown(f'<div class="selected-tab">{selected_option}</div>', unsafe_allow_html=True)

    if selected_option == "Upload an image":
        # st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.markdown("---")
            st.write("Classifying...")

            try:
                # Save the uploaded image to a local file
                image_path = "c:/Users/banal/Downloads/uploaded_image.jpg"
                with open(image_path, 'wb') as f:
                    f.write(uploaded_file.read())

                disease = identify_crop_disease(image_path)

                st.write(f"Identified Disease: {disease}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif selected_option == "Capture an image":
        # st.subheader("Capture an Image")
        st.write("Click the button below to capture an image:")
        if st.button("Capture Image"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            # Save the captured image to a local file
            image_path = "c:/Users/banal/Downloads/uploaded_image.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Display the captured image
            st.image(frame, caption='Captured Image.', use_column_width=True)
            st.markdown("---")
            st.write("Classifying...")

            try:
                disease = identify_crop_disease(image_path)

                st.write(f"Identified Disease: {disease}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    # Run the main function
    main()
