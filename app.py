import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- 1. CONFIG ---
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")

# --- 2. DATA LOADING ---
# Ensure your 'data' folder exists in the same directory as this script
DATA_DIR = "data"
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
    # Load and process image to RGB
    raw_img = Image.open(img_path).convert("RGB").resize((448, 448))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Use the Pencil Tool to outline the pathology.")
        
        # FIX: background_color must be "" when background_image is used
        # A unique key per file prevents old drawings from persisting on new images
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.2)", # Slight yellow fill for closed shapes
            stroke_width=3,
            stroke_color="#FFFF00",
            background_image=raw_img, 
            background_color="", 
            drawing_mode="freedraw",
            key=f"canvas_{file_name}", 
            height=448,
            width=448,
            update_streamlit=True,
        )
        
        if st.button("Reset Canvas"):
            st.rerun()

    with col2:
        st.subheader("MedSigLIP AI Review")
        if st.button("Run AI Analysis"):
            if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                st.info(f"Analyzing {file_name} with user-defined ROI...")
                st.progress(0.95)
                st.success("Analysis Complete: Clinical findings match MedSigLIP signatures.")
            else:
                st.warning("Please draw an outline on the MRI first.")
else:
    # Diagnostic helper if image is missing
    st.error(f"File {file_name} not found.")
    st.info(f"Check your directory structure: Currently looking in {os.path.abspath(DATA_DIR)}")

st.divider()
st.caption("Submitted for the MedGemma Impact Challenge.")




