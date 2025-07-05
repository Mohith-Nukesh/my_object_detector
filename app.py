import streamlit as st
import torch
from PIL import Image
import numpy as np

st.title("🧠 Object Detection with YOLOv5")

# Load model
@st.cache_resource
def load_model():
    return torch.load("yolov5s.pt", map_location=torch.device('cpu'))

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Inference
    results = model(image)
    results.render()
    st.image(results.ims[0], caption="Detected Objects", use_column_width=True)
