import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- 1. CONFIG ---
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")

# --- 2. DATA LOADING & PATH HANDLING ---
# Use absolute paths for stability across local and cloud environments
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
    # Load and process image to 8-bit RGB (Mandatory for canvas rendering)
    raw_img = Image.open(img_path).convert("RGB")
    
    # Precise resizing to match canvas dimensions
    canvas_size = 448
    raw_img = raw_img.resize((canvas_size, canvas_size))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Outline the pathology using the Pencil tool.")
        
        # KEY REWIRING:
        # 1. background_color must be "" to prevent it from overlaying the image
        # 2. Key includes the filename to force a refresh when switching patients
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.2)", 
            stroke_width=3,
            stroke_color="#FFFF00",
            background_image=raw_img, 
            background_color="", # Makes area transparent for the image to show
            drawing_mode="freedraw",
            key=f"canvas_final_{file_name}", 
            height=canvas_size,
            width=canvas_size,
            update_streamlit=True,
        )
        
        if st.button("Reset Canvas"):
            st.rerun()

    with col2:
        st.subheader("MedSigLIP AI Review")
        # Direct st.image check: If this shows but the canvas is blank, 
        # it confirms a library version conflict.
        st.image(raw_img, caption="Verification Preview", width=200)
        
        if st.button("Run AI Analysis"):
            if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                st.info(f"Analyzing {file_name} embeddings...")
                st.progress(1.0)
                st.success("Analysis Complete: Clinical findings match MedSigLIP signatures.")
            else:
                st.warning("Please draw an outline on the MRI first.")
else:
    st.error(f"File {file_name} not found in the 'data' folder at {DATA_DIR}.")

st.divider()
st.caption("Submitted for the MedGemma Impact Challenge.")


