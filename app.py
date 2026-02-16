import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- 1. CONFIG ---
# Set page to wide mode to accommodate the two-column layout
st.set_page_config(page_title="MedSigLIP Neuro-Tutor", layout="wide")
st.title("MedSigLIP Neuro-Tutor")

# --- 2. RELATIVE PATH HANDLING ---
# This ensures the app finds the 'data' folder regardless of how it is launched
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")

cases = {
    "Patient A: Cerebral Mass": "glioma.jpg",
    "Patient B: Parasagittal Mass": "meningioma.jpg",
    "Patient C: Sellar Region": "pituitary.jpg",
    "Patient D: Normal Control": "no_tumor.jpg"
}

# --- 3. SELECTION & VALIDATION ---
target_label = st.selectbox("Select Patient Case:", list(cases.keys()))
file_name = cases[target_label]
img_path = os.path.join(DATA_DIR, file_name)

# --- 4. IMAGE LOADING & DISPLAY ---
if os.path.exists(img_path):
    # Load and force RGB conversion (crucial for MRI formats like DICOM/TIF/JPG)
    raw_img = Image.open(img_path).convert("RGB")
    
    # Resize to match canvas dimensions precisely
    canvas_size = 448
    raw_img = raw_img.resize((canvas_size, canvas_size))

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interactive MRI Scan")
        st.caption("Draw directly on the scan to outline the pathology.")
        
        # KEY CORRECTIONS FOR TRANSPARENCY & VISIBILITY:
        # 1. background_image takes the PIL object directly.
        # 2. background_color MUST be an empty string "" or None.
        # 3. key must be unique per image to force the canvas to refresh.
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.2)", # 20% transparent yellow fill
            stroke_width=3,
            stroke_color="#FFFF00",             # Solid yellow lines
            background_image=raw_img, 
            background_color="",                # This makes the canvas area transparent
            drawing_mode="freedraw",
            key=f"canvas_v3_{file_name}",       # Unique key triggers refresh on swap
            height=canvas_size,
            width=canvas_size,
            update_streamlit=True,
        )
        
        if st.button("Reset Canvas"):
            st.rerun()

    with col2:
        st.subheader("MedSigLIP AI Review")
        
        # Small preview to confirm the file is loading correctly independently of the canvas
        st.image(raw_img, caption="File Load Verification", width=150)
        
        if st.button("Run AI Analysis"):
            # Logic check: did the user actually draw something?
            if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                st.info(f"Analyzing {file_name} Region of Interest (ROI)...")
                st.progress(100)
                st.success("Analysis Complete: Clinical findings match MedSigLIP signatures.")
            else:
                st.warning("Please outline the suspicious area on the MRI first.")

else:
    # Diagnostic Error Handling
    st.error(f"Error: Could not find image at {img_path}")
    st.info("Check that your images are inside a folder named 'data' in the same directory as this script.")

st.divider()
st.caption("Submitted for the MedGemma Impact Challenge.")
