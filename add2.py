import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="🌱 Plant Disease Prediction", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Prediction using CNN")
st.write("Upload a plant leaf image to predict its disease category using a trained CNN model.")

# --- Load the trained model ---
@st.cache_resource
def load_model():
    model_path = r"C:\Users\rajs1\Downloads\Ai_Project\plant_disease_prediction_model (1).h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please check the path and ensure the file exists.")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# --- Detect expected input shape ---
input_shape = model.input_shape[1:4]
expected_height, expected_width, _ = input_shape
st.write(f"🧩 Model expects input shape: {input_shape}")

# --- Helper function for preprocessing ---
def preprocess_image(image):
    img = image.resize((expected_width, expected_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Class Labels (38 PlantVillage classes) ---
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- File Upload Section ---
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Predicting disease... Please wait"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

    if predicted_class < len(class_names):
        st.success(f"🌾 **Prediction:** {class_names[predicted_class]}")
        st.info(f"🧠 **Confidence:** {confidence:.2f}%")
        st.bar_chart(prediction[0])
        
        # Optional download
        result_text = f"Prediction: {class_names[predicted_class]}, Confidence: {confidence:.2f}%"
        st.download_button("📥 Download Result", result_text, file_name="prediction.txt")
    else:
        st.error("⚠️ Prediction index out of range. Please verify the model and class list.")

# --- Footer ---
st.caption("Built with ❤️ using Streamlit and TensorFlow")
