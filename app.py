# === FILE: app.py ===
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognition", layout="centered")

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    return load_model("models/cnn_model.h5")

model = load_cnn_model()

st.title("✍️ Handwritten Digit Recognition")
st.markdown("Draw a digit (0–9) in the box below and click **Predict**")

# Drawing canvas setup
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict digit
if st.button("🧠 Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Threshold and find bounding box
        _, img_thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(img_thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit = img_gray[y:y+h, x:x+w]

            # Resize keeping aspect ratio
            aspect_ratio = w / h
            if aspect_ratio > 1:
                resized_digit = cv2.resize(digit, (20, int(20 / aspect_ratio)))
            else:
                resized_digit = cv2.resize(digit, (int(20 * aspect_ratio), 20))

            # Pad to 28x28
            pad_top = (28 - resized_digit.shape[0]) // 2
            pad_bottom = 28 - resized_digit.shape[0] - pad_top
            pad_left = (28 - resized_digit.shape[1]) // 2
            pad_right = 28 - resized_digit.shape[1] - pad_left
            img_resized = np.pad(resized_digit, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        else:
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert if needed
        if np.mean(img_resized) > 127:
            img_resized = 255 - img_resized

        img_normalized = img_resized.astype("float32") / 255.0
        img_input = img_normalized.reshape(1, 28, 28, 1)

        prediction = model.predict(img_input)
        predicted_digit = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: **{predicted_digit}**")
        st.image(img_resized, caption="Processed Input", width=100, clamp=True)
    else:
        st.warning("Please draw a digit first.")
