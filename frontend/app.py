import streamlit as st
import requests

st.title("YOLOv8m Object Detection (FastAPI + Streamlit UI)")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", width=400)

if st.button("Detect") and uploaded_image is not None:
    with st.spinner("Processing..."):
        files = {"file": uploaded_image.getvalue()}
        response = requests.post("http://localhost:8000/predict", files=files)
        result_path = response.json()["result_image"]
        result_url = f"http://localhost:8000/{result_path}"
        st.image(result_url, caption="Detected Output", width=500)
