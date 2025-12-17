import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Image Classification",
    layout="wide"
)

# =========================
# Constants
# =========================
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = (150, 150)

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_baseline.h5")
    tl  = tf.keras.models.load_model("transfer_learning_model.h5")
    return cnn, tl

cnn_model, tl_model = load_models()

# =========================
# Session State (History)
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Sidebar
# =========================
st.sidebar.title("Input")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["CNN Baseline", "Transfer Learning", "Compare Both"]
)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify",
    type=["jpg", "jpeg", "png"]
)

run_btn = st.sidebar.button("Classify Image")

# =========================
# Main Layout
# =========================
col1, col2 = st.columns([1.5, 1])

# =========================
# Image Section
# =========================
with col1:
    st.markdown("## Image Classification")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        st.caption(uploaded_file.name)
    else:
        st.info("Please upload an image from the left panel.")

# =========================
# Prediction Section
# =========================
with col2:
    if uploaded_file and run_btn:

        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        results = {}

        if model_choice in ["CNN Baseline", "Compare Both"]:
            results["CNN Baseline"] = cnn_model.predict(img_array)[0]

        if model_choice in ["Transfer Learning", "Compare Both"]:
            results["Transfer Learning"] = tl_model.predict(img_array)[0]

        for model_name, preds in results.items():

            top_idx = np.argmax(preds)
            top_class = CLASSES[top_idx]
            confidence = preds[top_idx]

            # Save history
            st.session_state.history.append(
                (model_name, top_class, confidence)
            )

            st.markdown(f"### {model_name}")
            st.write(f"**Prediction:** {top_class}")
            st.write(f"**Confidence:** {confidence:.4f}")

            # =========================
            # Probability Table
            # =========================
            df = pd.DataFrame({
                "Class": CLASSES,
                "Confidence": preds
            })

            df = df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)
            df.index += 1  # Start index from 1

            st.markdown("**Class Probabilities**")
            st.dataframe(
                df.style.format({"Confidence": "{:.4f}"}),
                use_container_width=True
            )

            # =========================
            # Bar Chart
            # =========================
            fig, ax = plt.subplots()
            ax.bar(df["Class"], df["Confidence"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence")
            ax.set_xlabel("Class")
            plt.xticks(rotation=30)

            st.pyplot(fig)

# =========================
# Prediction History
# =========================
if st.session_state.history:
    st.divider()
    st.markdown("### Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=["Model", "Predicted Class", "Confidence"]
    )

    history_df.index += 1
    st.dataframe(
        history_df.tail(10),
        use_container_width=True
    )

# =========================
# Footer
# =========================
st.divider()
st.markdown(
    "<center>Image Classification using CNN and Transfer Learning</center>",
    unsafe_allow_html=True
)
