import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os

# --- 1. Math Function ---
def calculate_dice(canvas_data, gt_mask):
    """Calculates overlap between drawing and ground truth."""
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. Page Config ---
st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# Filenames from your repo
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

# --- 3. Robust Image Loader ---
@st.cache_data
def get_processed_images(img_path, msk_path):
    """Loads and standardizes images for the canvas."""
    # Forced conversion to 8-bit RGB is mandatory for .tif files
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    msk = Image.open(msk_path).convert("RGB").resize((256, 256))
    return img, msk

if not os.path.exists(IMG_FILE):
    st.error(f"Cannot find {IMG_FILE} in repository root.")
else:
    bg_image, gt_mask = get_processed_images(IMG_FILE, MSK_FILE)

    st.subheader("1. Outline the tumor boundary")
    st.write("If you see a white box, try a 'Hard Refresh' (Ctrl+Shift+R).")

    # Use a unique key based on a counter to force re-renders
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    if st.button("Reset Canvas"):
        st.session_state.canvas_key += 1
        st.rerun()

    # THE CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image, # Passing processed PIL image
        drawing_mode="polygon",
        key=f"canvas_v{st.session_state.canvas_key}",
        height=256,
        width=256,
        update_streamlit=True
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            c1, c2 = st.columns(2)
            c1.metric("Dice Score", f"{score:.2%}")
            c2.image(gt_mask, caption="Expert Mask", use_container_width=True)
            
            if score > 0.85:
                st.success("Expert Level Match!")
            else:
                st.warning("Outline does not match expert ground truth.")
