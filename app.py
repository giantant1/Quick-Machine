import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="MedSigLIP Neuro‑Tutor", layout="wide")
st.title("MedSigLIP Neuro‑Tutor")
st.markdown("### Clinical Training: Brain Tumor Identification & Zero‑Shot Review")

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    st.error("Folder 'data/' not found. Create it and add MRI JPG/PNG files.")
    st.stop()

image_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not image_files:
    st.warning("No images found in 'data/'. Upload glioma.jpg, meningioma.jpg, etc.")
    st.stop()

# Sidebar
st.sidebar.header("Patient Database")
selected_file = st.sidebar.selectbox("Select Patient Case:", image_files)
img_path = os.path.join(DATA_DIR, selected_file)

# ---------------------------------------------------------
# 3. MAIN APP
# ---------------------------------------------------------
if os.path.exists(img_path):

    # Load image safely
    raw_img = Image.open(img_path).convert("RGB")

    # Create a resized copy for display
    display_img = raw_img.copy()
    display_img.thumbnail((448, 448))

    # Convert to NumPy for st_canvas
    display_np = np.array(display_img)

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # LEFT COLUMN — INTERACTIVE MRI CANVAS
    # -----------------------------------------------------
    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Use the pencil tool to highlight suspected pathology.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=5,
            stroke_color="#FFFFFF",
            background_image=display_np,     # FIXED
            drawing_mode="freedraw",
            key=f"canvas_{selected_file}",   # refreshes per case
            height=display_img.height,
            width=display_img.width,
            update_streamlit=True
        )

        if st.button("Reset Scan"):
            st.rerun()

    # -----------------------------------------------------
    # RIGHT COLUMN — AI ANALYSIS
    # -----------------------------------------------------
    with col2:
        st.subheader("MedSigLIP AI Review")
        st.write("Run the simulated zero‑shot classifier.")

        if st.button("Run AI Analysis"):
            st.info("Generating MedSigLIP Image Embeddings...")

            # Simple simulation for demo
            fname = selected_file.lower()

            if "glioma" in fname:
                scores = {"Glioma": 0.96, "Meningioma": 0.02, "Other": 0.02}
            elif "meningioma" in fname:
                scores = {"Meningioma": 0.91, "Glioma": 0.07, "Other": 0.02}
            elif "pituitary" in fname:
                scores = {"Pituitary": 0.94, "Other": 0.06}
            else:
                scores = {"Pathology": "Standard Clinical Signature Detected"}

            # Display results
            if isinstance(scores, dict):
                for label, val in scores.items():
                    st.write(f"**{label}**")
                    if isinstance(val, float):
                        st.progress(val)
                    else:
                        st.write(val)

            st.success("Analysis Complete")
            st.markdown("**Educational Insight:** MedSigLIP identifies visual tokens associated with clinical reports.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("Built for Medical Education • MedSigLIP Neuro‑Tutor")


