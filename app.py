import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os
from huggingface_hub import hf_hub_download

# Hugging Face model details
MODEL_REPO = "yuvinraja/image-forgery-model"
MODEL_FILENAME = "model_casia_run1.h5"

# Function to download model
def download_model():
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading model from Hugging Face...")
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=".")
        st.success("Model downloaded successfully!")

# Load model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_FILENAME, compile=False)

# Initialize model
model = load_model()
st.write("Model Loaded Successfully! ✅")

# Define the ELA function
def convert_to_ela_image(image, quality=91):
    temp_filename = "temp_file.jpg"
    ela_filename = "temp_ela.png"

    image = image.convert("RGB")
    image.save(temp_filename, "JPEG", quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# # Load the trained model
# MODEL_PATH = "model_casia_run1.h5"
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH, compile=False)

# model = load_model()

# Image preprocessing function
def prepare_image(image):
    ela_image = convert_to_ela_image(image, 91)
    ela_image_resized = ela_image.resize((128, 128))
    image_array = np.array(ela_image_resized) / 255.0  # Normalize
    image_array = image_array.reshape(-1, 128, 128, 3)  # Reshape for model
    return image_array

# Streamlit UI
st.title("🔍 Image Forgery Detection using CNN")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to ELA and display
    ela_image = convert_to_ela_image(image)
    st.image(ela_image, caption="ELA Image", use_container_width=True)

    # Process image and make prediction
    processed_image = prepare_image(image)
    predictions = model.predict(processed_image)
    class_names = ["fake", "real"]

    index = np.argmax(predictions[0])
    predicted_class = class_names[index]
    confidence = float(np.max(predictions[0]))  # Convert to float

    # Display prediction
    st.write(f"### Prediction: **{predicted_class.upper()}**")
    st.write(f"### Confidence Score: {confidence:.2f}")

    # Show status message
    if predicted_class == "fake":
        st.error("❌ This image is likely forged!")
    else:
        st.success("✅ This image appears authentic!")

