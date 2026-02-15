import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os

# --- 1. Helper Functions ---

def calculate_dice(canvas_data, gt_mask):
    """Calculates Dice Similarity Coefficient (DSC)."""
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = (gt_array > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. App Interface ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")
st.info("BraTS Standard: Use the **Polygon Tool** to trace the tumor boundary.")

# UPDATED FILENAMES based on your repo
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

# Debugging: Check if files exist
if not os.path.exists(IMG_FILE) or not os.path.exists(MSK_FILE):
    st.error("Missing files! One or both files were not found in the repository root.")
    st.write("Files actually found in your repo:", os.listdir("."))
else:
    try:
        # Load and convert for browser display
        bg_image = Image.open(IMG_FILE).convert("RGB")
        gt_mask = Image.open(MSK_FILE).convert("RGB")

        st.subheader("1. Outline the tumor boundary")
        if st.button("Reset Canvas"): 
            st.rerun()

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

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            st.divider()
            choice = st.radio("Suspected grade:", ["LGG", "HGG", "Meningioma"])
            
            if st.button("Submit My Answer"):
                score = calculate_dice(canvas_result.image_data, gt_mask)
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Your Dice Score", f"{score:.2%}")
                    st.write("**Expert Level:** > 0.85")
                    if score > 0.85:
                        st.success(f"Expert Match! Clinical: {choice}")
                    else:
                        st.warning("Outline mismatch. Compare with the expert mask.")
                
                with c2:
                    st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing medical images: {e}")
