import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pandas as pd

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
train_csv = pd.read_csv('Train.csv')
test_csv = pd.read_csv('Test.csv')
train_csv['ClassId'] = train_csv['ClassId'].astype(str)
test_csv['ClassId'] = test_csv['ClassId'].astype(str)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.6, 1.4],  # More brightness variation
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest',
    rescale=1.0 / 255.0,  # Normalization
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    x_col='Path',
    y_col='ClassId',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

class_labels = {str(v): k for k, v in train_generator.class_indices.items()}
with open("class_labels.json", "w") as f:
    json.dump(class_labels, f)
print("Saved corrected class_labels.json!")


# Load actual class labels from the dataset
with open('class_labels.json', 'r') as f:
    CLASS_NAMES = json.load(f)
print("Class indices mapping in model:", test_generator.class_indices)
print("Loaded class labels:", CLASS_NAMES)



MODEL_PATH = 'traffic_sign_model_finetuned.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)  
    image = image.convert('RGB')  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit app UI
st.title("Traffic Sign Classification")
st.write("Upload an image of a traffic sign, and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)  
    confidence = np.max(predictions) * 100  

    # **Fix: Use correct mapping for class names**
    predicted_label = CLASS_NAMES.get(str(predicted_class), "Unknown")

    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
