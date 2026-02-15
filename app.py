import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os

# --- 1. Math Function ---
def calculate_dice(canvas_data, gt_mask):
    """Calculates Dice Similarity Coefficient (DSC)."""
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. Page Setup ---
st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# FILENAMES
IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

if not os.path.exists(IMG_FILE):
    st.error(f"Image '{IMG_FILE}' not found! Check your main branch.")
else:
    # Load and process images
    # Using .convert("L").convert("RGB") helps some .tif files render better
    bg_image = Image.open(IMG_FILE).convert("RGB").resize((256, 256))
    gt_mask = Image.open(MSK_FILE).convert("RGB").resize((256, 256))

    st.subheader("1. Outline the tumor boundary")
    st.write("Click to create points. Double-click or click the first point to close the polygon.")

    # Using a session state counter for the key ensures the canvas refreshes
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    if st.button("Reset Canvas"):
        st.session_state.canvas_key += 1
        st.rerun()

    # THE CANVAS - Passing the PIL image object directly
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key=f"canvas_{st.session_state.canvas_key}",
        height=256,
        width=256,
        update_streamlit=True
    )

    # Submission Logic
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Dice Score", f"{score:.2%}")
                if score > 0.85:
                    st.success("Expert Level Match!")
                else:
                    st.warning("Try refining your outline.")
            with col2:
                st.image(gt_mask, caption="Expert Mask", use_container_width=True)


