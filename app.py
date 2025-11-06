import streamlit as st
st.set_page_config(page_title="DeepFake Detector", page_icon="🧠", layout="centered")

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===============================
# 🧩 Load model (cached for speed)
# ===============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("deepfake_efficientnet_b3_best.keras")
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ===============================
# 🧠 App Header
# ===============================
st.title("🧠 DeepFake Detection:See beyond deception")
st.write("Upload an image to check whether it’s **Real** or **Fake** ")

# ===============================
# 📤 File Upload Section
# ===============================
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess exactly like in training
    img_resized = image.resize((300, 300))  # EfficientNet-B3 input size
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)  # ✅ correct preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # 🔍 Run Prediction
    # ===============================
    with st.spinner("Analyzing image... 🔎"):
        pred = model.predict(img_array, verbose=0)[0][0]
        # Default: fake=0, real=1 (alphabetical order)
        label = "🟢 Real" if pred > 0.5 else "🔴 Fake"
        confidence = pred if pred > 0.5 else 1 - pred

    # ===============================
    # 📊 Display Results
    # ===============================
    st.subheader("🧾 Prediction Results")
    st.metric(label="Prediction", value=label)
    st.progress(float(confidence))
    st.caption(f"Confidence: **{confidence*100:.2f}%**")

    st.info(
        "ℹ️ Note: This model was trained on deepfake datasets. "
        "For AI-generated (GAN / diffusion) images, accuracy may vary."
    )

else:
    st.warning("Please upload an image to begin the analysis.")
