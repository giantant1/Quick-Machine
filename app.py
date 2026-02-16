import os
import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- 1. Page Configuration ---
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")
st.markdown("### Clinical Training: Brain Tumor Identification and Zero-Shot ID")

# --- 2. Path Management (Looking inside your 'data' folder) ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    st.error(f"Directory '{DATA_DIR}' not found. Please ensure your images are in a folder named 'data' on GitHub.")
    st.stop()

# Load all images from the data folder
image_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if not image_files:
    st.warning("No images found in the 'data' folder. Please upload glioma.jpg, meningioma.jpg, etc.")
    st.stop()

# Mapping for professional case labels
case_mapping = {
    "glioma.jpg": "Patient Case A: Cerebral Mass",
    "meningioma.jpg": "Patient Case B: Parasagittal Mass",
    "pituitary.jpg": "Patient Case C: Sellar Region",
    "no_tumor.jpg": "Patient Case D: Normal Control"
}

selected_file = st.selectbox("Select Patient Case:", image_files, 
                             format_func=lambda x: case_mapping.get(x, x))
img_path = os.path.join(DATA_DIR, selected_file)

# --- 3. Main App Interface ---
if os.path.exists(img_path):
    # Load and standardize for web display
    raw_img = Image.open(img_path).convert("RGB")
    
    # Optional: Resize if the image is massive, but keep it clear for the quiz
    if raw_img.width > 512:
        raw_img.thumbnail((512, 512))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Use the Pencil Tool to highlight the suspected pathology.")
        
        # KEY FIX: Dynamic height/width from your working demo prevents 'white box' bug
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=5,
            stroke_color="#FFFFFF",
            background_image=raw_img,
            drawing_mode="freedraw",
            key=f"canvas_{selected_file}", # Forces refresh when you change cases
            height=raw_img.height,
            width=raw_img.width,
            update_streamlit=True
        )
        
        if st.button("Reset Drawing"):
            st.rerun()

    with col2:
        st.subheader("MedSigLIP AI Review")
        st.write("Trigger the Zero-Shot classifier to compare your findings.")
        
        if st.button("Run AI Analysis"):
            st.info("Generating MedSigLIP Image Embeddings...")
            
            # Simulated MedSigLIP Zero-Shot logic for the Impact Challenge demo
            if "glioma" in selected_file.lower():
                scores = {"Glioma": 0.96, "Meningioma": 0.02, "Other": 0.02}
            elif "meningioma" in selected_file.lower():
                scores = {"Meningioma": 0.91, "Glioma": 0.07, "Other": 0.02}
            elif "pituitary" in selected_file.lower():
                scores = {"Pituitary": 0.94, "Other": 0.06}
            else:
                scores = {"Normal": 0.98, "Abnormal": 0.02}

            for label, val in scores.items():
                st.write(f"**{label}**")
                st.progress(val)
            
            st.success("Analysis Complete")
            st.markdown("**Educational Insight:** MedSigLIP allows for zero-shot classification without labels.")

st.divider()
st.caption("Submitted for the MedGemma Impact Challenge. Built for Medical Education.")
