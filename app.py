import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")
st.markdown("### Clinical Training: Brain Tumor Identification & Zero-Shot ID")

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    st.error("Directory 'data' not found. Create a folder named 'data' and upload your JPG/PNG MRI scans.")
    st.stop()

image_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if not image_files:
    st.warning("No images found in 'data'. Please upload files like glioma.jpg, meningioma.jpg, pituitary.jpg, etc.")
    st.stop()

st.sidebar.header("Patient Database")
selected_file = st.sidebar.selectbox("Select Patient Case:", image_files)
img_path = os.path.join(DATA_DIR, selected_file)

# ---------------------------------------------------------
# 3. MAIN APP LOGIC
# ---------------------------------------------------------
if os.path.exists(img_path):
    # Load and standardize for web display
    raw_img = Image.open(img_path).convert("RGB")

    # Create a resized copy for display
    display_img = raw_img.copy()
    display_img.thumbnail((448, 448))

    # Convert to NumPy for st_canvas
    display_np = np.array(display_img)

    col1, col2 = st.columns(2)

    # ---------------- LEFT: INTERACTIVE MRI ----------------
    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Use the Pencil Tool to highlight the suspected pathology.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=5,
            stroke_color="#FFFFFF",
            background_image=display_np,
            drawing_mode="freedraw",
            key=f"canvas_{selected_file}",
            height=display_img.height,
            width=display_img.width,
            update_streamlit
