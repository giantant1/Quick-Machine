import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

# --- 1. Math Function ---
def calculate_dice(canvas_data, gt_mask):
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. Image to Base64 (The Secret Sauce) ---
def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG") # PNG is safest for canvas
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# EXACT FILENAMES
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

if not os.path.exists(IMG_FILE):
    st.error("Image file missing from repo root!")
else:
    # Load and process
    raw_img = Image.open(IMG_FILE).convert("RGB").resize((256, 256))
    gt_mask = Image.open(MSK_FILE).convert("RGB").resize((256, 256))

    # Convert to Base64 to force it to show in the canvas
    bg_base64 = get_image_base64(raw_img)

    st.subheader("1. Outline the tumor")
    
    # Use the base64 string as the background_image
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=raw_img, # Pass PIL image, but background_color may interfere
        drawing_mode="polygon",
        key="canvas",
        height=256,
        width=256,
    )

    if st.button("Reset"): st.rerun()

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Dice Score", f"{score:.2%}")
            c2.image(gt_mask, caption="Expert Mask")


