import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

# --- 1. Helper Functions ---

def calculate_dice(canvas_data, gt_mask):
    """
    Calculates Dice Similarity Coefficient (DSC).
    Compares the student's drawing to the expert mask.
    """
    # Student mask: 1 where drawn (alpha channel > 0)
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    
    # Expert mask: 1 where pixel > 0
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = (gt_array > 0).astype(np.uint8)
    
    # Math: (2 * intersection) / (total pixels)
    intersection = np.sum(student_mask * gt_binary)
    dice = (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)
    return dice

# --- 2. App Interface ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")
st.info("BraTS Standard: Use the **Polygon Tool** to trace the tumor boundary.")

# UPDATE THESE FILENAMES to match exactly what you uploaded to GitHub
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_1_mask.tif"

try:
    # Load .tif files locally from your repository
    # .convert("RGB") is critical for .tif files to display in browsers
    bg_image = Image.open(IMG_FILE).convert("RGB")
    gt_mask = Image.open(MSK_FILE).convert("RGB")

    st.subheader("1. Outline the tumor boundary")
    
    if st.button("Reset Canvas"):
        st.rerun()

    # Drawing Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key="canvas",
        height=256,
        width=256,
    )

    # Submission Logic
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        st.subheader("2. Final Diagnosis")
        choice = st.radio("Based on the scan, what is the grade?", ["LGG", "HGG", "Meningioma"])
        
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Dice Score", f"{score:.2%}")
                st.write("**Expert Level:** > 0.85")
                if score > 0.85:
                    st.success(f"Excellent! Matches clinical {choice} data.")
                else:
                    st.warning("Outline mismatch. Compare with the expert mask.")
            
            with col2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)

except FileNotFoundError:
    st.error(f"Files not found! Make sure **{IMG_FILE}** and **{MSK_FILE}** are uploaded to your GitHub repository.")
