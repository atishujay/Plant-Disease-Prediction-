import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
# ✅ Direct model & json file path (no need for __file__)
MODEL_PATH = "plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)   # Resize
    img_array = np.array(img)         # Convert to numpy
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype("float32") / 255.0  # Normalize
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title("🌱 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button("Classify"):
            raw_prediction = predict_image_class(model, image, class_indices)

            # Clean name
            pretty_prediction = raw_prediction.replace("_", " ")
            parts = pretty_prediction.split()

            if len(parts) > 1 and parts[0].lower() == parts[1].lower():
                plant = parts[0]
                disease = " ".join(parts[1:])
            else:
                # Fallback if label is simple
                plant = parts[0]
                disease = " ".join(parts[1:]) if len(parts) > 1 else "Healthy"

            st.success(f"🦠 Disease: {disease}")

        