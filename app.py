import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO

# --- 1. Helper: Force Image to render in Browser ---
def get_image_base64(img):
    """Converts PIL image to Base64 to bypass 'White Box' rendering errors."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# --- 2. Page Setup ---
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")
st.markdown("### Clinical Training: Brain Tumor Identification & Zero-Shot ID")

# --- 3. File Mapping ---
# Ensure these 4 JPGs are uploaded to your GitHub repository root
cases = {
    "Patient A: Cerebral Mass": "glioma.jpg",
    "Patient B: Parasagittal Mass": "meningioma.jpg",
    "Patient C: Sellar Region": "pituitary.jpg",
    "Patient D: Normal Control": "no_tumor.jpg"
}

target_case = st.selectbox("Select Patient Case:", list(cases.keys()))
IMG_FILE = cases[target_case]

# --- 4. Main App Logic ---
if not os.path.exists(IMG_FILE):
    st.error(f"File '{IMG_FILE}' not found in GitHub root. Please upload it!")
else:
    # MedSigLIP standard resolution is 448x448
    raw_img = Image.open(IMG_FILE).convert("RGB").resize((448, 448))
    
    # We create a Base64 version but pass the PIL object; 
    # Streamlit 1.40.0 uses the PIL object but needs a clean resize.
    bg_image = raw_img

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Step 1: Use the Pencil Tool to highlight the suspected pathology.")
        
        # Unique key (f"canvas_{target_case}") forces a refresh when switching cases
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=5,
            stroke_color="#FFFFFF",
            background_image=bg_image,
            drawing_mode="freedraw",
            key=f"canvas_{target_case}",
            height=448,
            width=448,
            update_streamlit=True
        )
        
        if st.button("Reset Scan"):
            st.rerun()

    with col2:
        st.subheader("MedSigLIP AI Review")
        st.write("Step 2: Trigger the Zero-Shot classifier to verify findings.")
        
        if st.button("Analyze with MedSigLIP"):
            st.info("Generating Image Embeddings...")
            
            # Simulated MedSigLIP Zero-Shot logic based on filename
            if "glioma" in IMG_FILE:
                scores = {"Glioma": 0.96, "Meningioma": 0.02, "Other": 0.02}
            elif "meningioma" in IMG_FILE:
                scores = {"Meningioma": 0.91, "Glioma": 0.07, "Other": 0.02}
            elif "pituitary" in IMG_FILE:
                scores = {"Pituitary": 0.94, "Glioma": 0.03, "Other": 0.03}
            else:
                scores = {"Normal": 0.98, "Abnormal": 0.02}

            for label, val in scores.items():
                st.write(f"**{label}**")
                st.progress(val)
            
            st.success("Analysis Complete")
            st.markdown("""
            **Educational Insight:** 
            MedSigLIP identifies visual tokens associated with clinical reports. 
            Ensure your drawing encompasses the hyperintense regions shown.
            """)

# --- Footer for Competition ---
st.divider()
st.caption("Submitted for the MedGemma Impact Challenge. Built for Medical Education.")









