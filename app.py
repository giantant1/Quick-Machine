import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import os

# --- 1. Math ---
def calculate_dice(canvas_data, gt_mask):
    """Calculates overlap between pencil drawing and ground truth."""
    # Student mask: 1 where pencil marks exist (alpha > 0)
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    # Expert mask: 1 where pixel > 0
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    intersection = np.sum(student_mask * gt_binary)
    return (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)

# --- 2. App Logic ---
st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

IMG_FILE = "TCGA_CS_4944_20010208_1.tif"
MSK_FILE = "TCGA_CS_4944_20010208_10_mask.tif"

if not os.path.exists(IMG_FILE):
    st.error(f"Cannot find {IMG_FILE}. Verify it is in your main branch.")
else:
    # PROCESS IMAGE: Force 16-bit TIF into 8-bit RGB for web display
    raw_tif = Image.open(IMG_FILE)
    bg_image = ImageOps.autocontrast(raw_tif.convert("L")).convert("RGB").resize((256, 256))
    
    # Process mask
    gt_mask = Image.open(MSK_FILE).convert("RGB").resize((256, 256))

    st.subheader("1. Use the Pencil to color in the tumor")
    st.caption("Click and drag to draw. Use the 'Reset' button to clear.")

    # THE CANVAS (Freedraw Mode)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=5,         # Thicker line for easier pencil drawing
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="freedraw", # UPDATED: Changed from polygon to freedraw
        key="pencil_v1",
        height=256,
        width=256,
        update_streamlit=True,
    )

    if st.button("Reset Drawing"):
        st.rerun()

    # Check if anything was drawn
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Dice Similarity Score", f"{score:.2%}")
                if score > 0.80:
                    st.success("Great job segmenting the tumor!")
                else:
                    st.warning("Try to be more precise with your pencil.")
            with c2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)



