import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import os

# --- 1. Math ---
def calculate_dice(canvas_data, gt_mask):
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. App Logic ---
st.title("Quick Machine: Tumor Segmentation Quiz")

IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

if not os.path.exists(IMG_FILE):
    st.error(f"Cannot find {IMG_FILE} in your main branch.")
else:
    # CRITICAL FIX: Convert 16-bit TIF to 8-bit RGB for the web
    raw_tif = Image.open(IMG_FILE)
    # Normalize and convert to standard 8-bit RGB
    bg_image = ImageOps.autocontrast(raw_tif.convert("L")).convert("RGB").resize((256, 256))
    
    # Process mask the same way
    raw_msk = Image.open(MSK_FILE)
    gt_mask = raw_msk.convert("RGB").resize((256, 256))

    st.subheader("1. Outline the tumor boundary")

    # THE CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key="final_v3",
        height=256,
        width=256,
        update_streamlit=True,
    )

    if st.button("Reset"): st.rerun()

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Dice Score", f"{score:.2%}")
            c2.image(gt_mask, caption="Expert Mask")


