import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib
from keras_facenet import FaceNet

# Load trained model and label dictionary
model = load_model("face_recognition_cnn.h5")
label_dict = joblib.load("label_dict.pkl")

# Load FaceNet for feature extraction
facenet_model = FaceNet()

# Function to extract face and compute FaceNet embeddings
def extract_features(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None  # No face detected

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))  # Resize to 160x160 for FaceNet
    face = np.expand_dims(face, axis=0)  # Add batch dimension

    # Extract FaceNet embeddings
    features = facenet_model.embeddings(face)[0]
    return np.expand_dims(features, axis=0)  # Expand dims for model input

# Streamlit UI
st.title("🔍 Face Recognition App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract FaceNet features
    features = extract_features(img)

    if features is not None:
        # Predict
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Convert label index to actual name
        person_name = [name for name, idx in label_dict.items() if idx == predicted_label][0]

        st.success(f"✅ Prediction: **{person_name}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("⚠️ No face detected. Try another image.")
