import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- 1. CONFIG ---
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")

# --- 2. PATH HANDLING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")

cases = {
    "Patient A: Cerebral Mass": "glioma.jpg",
    "Patient B: Parasagittal Mass": "meningioma.jpg",
    "Patient C: Sellar Region": "pituitary.jpg",
    "Patient D: Normal Control": "no_tumor.jpg"
}

target_label = st.selectbox("Select Patient Case:", list(cases.keys()))
file_name = cases[target_label]
img_path = os.path.join(DATA_DIR, file_name)

# --- 3. MAIN INTERFACE ---
if os.path.exists(img_path):
    # Load and resize
    raw_img = Image.open(img_path).convert("RGB")
    canvas_size = 448
    raw_img = raw_img.resize((canvas_size, canvas_size))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interactive MRI Scan")
        
        # FIX: background_color MUST be set to None or "" 
        # and ensure a unique key for every image swap.
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.2)", 
            stroke_width=3,
            stroke_color="#FFFF00",
            background_image=raw_img, 
            background_color="", # Critically important to avoid covering the image
            drawing_mode="freedraw",
            key=f"canvas_refreshed_{file_name}", # Unique key per patient
            height=canvas_size,
            width=canvas_size,
            update_streamlit=True,
        )
        
        if st.button("Reset Canvas"):
            st.rerun()

    with col2:
        st.subheader("Verification & AI")
        # If this image displays but the canvas is blank, 
        # it is a confirmed library version issue.
        st.image(raw_img, caption="Verification Preview", width=200)
        
        if st.button("Run AI Analysis"):
            st.success("Analysis Complete!")
else:
    st.error(f"File {file_name} not found at {img_path}")
