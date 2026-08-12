import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier.keras")


model = load_model()

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = {
    0: "Organic",
    1: "Recyclable"
}

# ==========================================
# HEADER
# ==========================================

st.title("♻️ Waste Classification AI")

st.write(
    "Upload an image of waste and the Deep Learning model "
    "will classify it as Organic or Recyclable."
)

st.divider()

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # Resize image
    image_resized = image.resize((128, 128))

    # Convert image to numpy array
    image_array = np.array(image_resized)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # ==========================================
    # DETERMINE CLASS
    # ==========================================

    if prediction >= 0.5:
        predicted_class = "Recyclable"
        confidence = prediction
    else:
        predicted_class = "Organic"
        confidence = 1 - prediction

    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    st.subheader("Prediction")

    st.success(
        f"♻️ {predicted_class}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    # ==========================================
    # PROBABILITY
    # ==========================================

    st.subheader("Prediction Confidence")

    organic_probability = (1 - prediction) * 100
    recyclable_probability = prediction * 100

    st.write(
        f"🌱 Organic: **{organic_probability:.2f}%**"
    )

    st.progress(
        int(organic_probability)
    )

    st.write(
        f"♻️ Recyclable: **{recyclable_probability:.2f}%**"
    )

    st.progress(
        int(recyclable_probability)
    )

else:

    st.info(
        "Please upload an image to get a prediction."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Waste Classification AI | "
    "Deep Learning CNN Project"
)