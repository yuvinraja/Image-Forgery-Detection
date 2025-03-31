import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os
from huggingface_hub import hf_hub_download
import io

# Hugging Face model details
MODEL_REPO = "yuvinraja/image-forgery-model"
MODEL_FILENAME = "model_casia_run1.h5"

# Sidebar with instructions
st.sidebar.title("ℹ️ How to Use")
st.sidebar.write("""
1️⃣ Upload an image (JPG/PNG).  
2️⃣ The app generates the **ELA image**.  
3️⃣ The **model predicts** whether the image is **Real** or **Fake**.  
4️⃣ Download the **ELA image** for reference.  
""")

# Function to download model
def download_model():
    model_path = os.path.join(".", MODEL_FILENAME)
    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model from Hugging Face..."):
            hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=".")
        st.success("✅ Model downloaded successfully!")

# Load model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_FILENAME, compile=False)

# Initialize model
model = load_model()
st.success("✅ Model Loaded Successfully!")

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

# Image preprocessing function
def prepare_image(image):
    ela_image = convert_to_ela_image(image, 91)
    ela_image_resized = ela_image.resize((128, 128))
    image_array = np.array(ela_image_resized) / 255.0  # Normalize
    image_array = image_array.reshape(-1, 128, 128, 3)  # Reshape for model
    return image_array, ela_image

# Streamlit UI
st.title("🔍 Image Forgery Detection using CNN")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_container_width=True)

    # Convert image to ELA
    with st.spinner("🔄 Generating ELA Image..."):
        processed_image, ela_image = prepare_image(image)
        st.image(ela_image, caption="📌 ELA Image", use_container_width=True)

    # Add a progress bar
    progress = st.progress(0)

    # Make prediction
    with st.spinner("🔄 Making Prediction..."):
        predictions = model.predict(processed_image)
        class_names = ["Fake", "Real"]

        confidence_fake = float(predictions[0][0]) * 100
        confidence_real = float(predictions[0][1]) * 100

        predicted_class = class_names[np.argmax(predictions[0])]

        # Update progress bar
        progress.progress(100)

    # Display prediction
    st.write(f"### 🎯 Prediction: **{predicted_class.upper()}**")
    st.write(f"**Confidence Score:**")
    st.write(f"- **Fake:** {confidence_fake:.2f}%")
    st.write(f"- **Real:** {confidence_real:.2f}%")

    # Show status message
    if predicted_class == "Fake":
        st.error("❌ This image is likely forged!")
    else:
        st.success("✅ This image appears authentic!")

    # Download ELA Image
    ela_buffer = io.BytesIO()
    ela_image.save(ela_buffer, format="PNG")
    st.download_button(
        label="📥 Download ELA Image",
        data=ela_buffer.getvalue(),
        file_name="ELA_Image.png",
        mime="image/png"
    )
