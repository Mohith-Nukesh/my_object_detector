import torch
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

model = load_model()

st.title("🧠 Object Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    results.render()
    st.image(results.ims[0], caption="Detected Objects", use_column_width=True)
