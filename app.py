import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define constants
IMG_SIZE = (64, 64)  # Image dimensions
CLASS_NAMES = [str(i) for i in range(43)]  
MODEL_PATH = 'D:/GUVI/Image Claasification/traffic_sign_model_finetuned.h5'  

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Function to preprocess uploaded image
def preprocess_image(image):
    """
    Preprocesses the uploaded image for the model.
    - Resizes the image to match the model input size.
    - Normalizes pixel values to the range [0, 1].
    - Expands dimensions to create a batch of size 1.
    """
    image = image.resize(IMG_SIZE)  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app UI
st.title("Traffic Sign Classification")
st.write("Upload an image of a traffic sign, and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability
    confidence = np.max(predictions) * 100  # Get confidence score

    # Display prediction and confidence
    st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
