import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

# --- 1. Math & Encoding ---
def calculate_dice(canvas_data, gt_mask):
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

def get_base64_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- 2. App UI ---
st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

if not os.path.exists(IMG_FILE):
    st.error("Image not found in main branch!")
else:
    # Process images
    raw_bg = Image.open(IMG_FILE).convert("RGB").resize((256, 256))
    gt_mask = Image.open(MSK_FILE).convert("RGB").resize((256, 256))
    
    # NEW: Create a background color if the image fails
    # This helps debug if the canvas itself is loading
    
    st.subheader("1. Outline the tumor boundary")
    
    # THE CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=raw_bg, # Passing PIL directly again with a fresh resize
        drawing_mode="polygon",
        key="final_v1",
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

