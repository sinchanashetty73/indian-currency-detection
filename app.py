import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Indian Currency Detection",
    layout="centered"
)

st.title("💵 Indian Currency Detection")
st.write("Upload or capture an image of an Indian currency note or coin")

# =========================
# Load Model (SAFE)
# =========================
@st.cache_resource
def load_currency_model():
    return tf.keras.models.load_model(
        "model/indian_currency_model.keras",
        compile=False
    )

model = load_currency_model()

# =========================
# Class Names
# (Must match training order)
# =========================
class_names = [
    "1_rupee_coin",
    "2_rupees_coin",
    "5_rupees_coin",
    "10_rupees_coin",
    "10_rupees_note",
    "20_rupees_note",
    "50_rupees_note",
    "100_rupees_note",
    "200_rupees_note",
    "500_rupees_note"
]

# =========================
# Image Preprocessing
# =========================
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# Choose Input Method
# =========================
option = st.radio(
    "Choose input method:",
    ("Upload Image", "Use Webcam")
)

image = None

# =========================
# Upload Image Option
# =========================
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload currency image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# =========================
# Webcam Option
# =========================
elif option == "Use Webcam":
    camera_image = st.camera_input("Capture currency image")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# =========================
# Prediction
# =========================
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    if class_id < len(class_names):
        st.success(
            f"💰 Prediction: **{class_names[class_id].replace('_', ' ').upper()}**\n\n"
            f"📊 Confidence: **{confidence:.2f}%**"
        )
    else:
        st.error("Prediction error: class index mismatch")
