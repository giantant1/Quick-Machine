import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os

# --- 1. Math Function ---
def calculate_dice(canvas_data, gt_mask):
    """Calculates Dice Similarity Coefficient (DSC)."""
    # Student mask: 1 where drawn (alpha channel > 0)
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    # Expert mask: 1 where pixel > 0
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    
    intersection = np.sum(student_mask * gt_binary)
    dice = (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)
    return dice

# --- 2. Page Setup ---
st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")
st.info("BraTS Standard: Use the **Polygon Tool** to trace the tumor boundary.")

# EXACT FILENAMES FROM YOUR REPO
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

# --- 3. Image Processing ---
if not os.path.exists(IMG_FILE):
    st.error(f"File '{IMG_FILE}' not found in main branch!")
    st.write("Files found:", os.listdir("."))
else:
    try:
        # Load and FORCE conversion to 8-bit RGB to prevent 'white box'
        raw_img = Image.open(IMG_FILE)
        bg_image = raw_img.convert("RGB").resize((256, 256))
        
        # Load mask and resize to match canvas exactly
        raw_mask = Image.open(MSK_FILE)
        gt_mask = raw_mask.convert("RGB").resize((256, 256))

        # --- 4. Interactive Canvas ---
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
            update_streamlit=True
        )

        # --- 5. Submission Logic ---
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            st.divider()
            choice = st.radio("Suspected Grade:", ["LGG", "HGG", "Meningioma"])
            
            if st.button("Submit My Answer"):
                score = calculate_dice(canvas_result.image_data, gt_mask)
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Your Dice Score", f"{score:.2%}")
                    st.write("**Expert Level:** > 0.85")
                    if score > 0.85:
                        st.success(f"Expert Match! Clinical: {choice}")
                    else:
                        st.warning("Outline mismatch. Compare with the expert mask.")
                
                with col2:
                    st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing images: {e}")

