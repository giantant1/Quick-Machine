import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(image_url):
    """
    Fetches the image directly from the Mateusz Buda GitHub assets.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status() 
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to reach GitHub: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """Calculates the overlap (Dice Score) between student and expert."""
    student_mask = np.where(canvas_data[:,:,3] > 0, 1, 0)
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = np.where(gt_array > 0, 1, 0)
    
    intersection = np.logical_and(student_mask, gt_binary).sum()
    dice = (2. * intersection) / (student_mask.sum() + gt_binary.sum() + 1e-7)
    return dice

# --- 2. App Logic ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")
st.info("BraTS Standard: Use the polygon tool to outline the tumor core.")

# DIRECT LINKS to the Buda Repo Assets
IMAGE_URL = "https://raw.githubusercontent.com"
MASK_URL = "https://raw.githubusercontent.com"

# Fetch images
bg_image = load_from_repo(IMAGE_URL)
gt_mask = load_from_repo(MASK_URL)

if bg_image:
    st.subheader("1. Outline the tumor boundary")
    
    if st.button("Clear Drawing"):
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
        display_toolbar=True
    )

    # Logic to move from Stage 1 to 2
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        st.subheader("2. Classify the pathology")
        choice = st.radio("What is your diagnosis?", 
                          ["LGG (Lower-Grade Glioma)", "HGG (High-Grade Glioma)", "Meningioma"])
        
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Dice Score", f"{score:.2%}")
                st.write("Expert threshold: > 0.85")
            
            with col2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
            
            if score > 0.80:
                st.success(f"Excellent! Your identification as {choice} matches clinical data.")
            else:
                st.warning("Your segmentation boundary needs adjustment.")
else:
    st.warning("Waiting for medical images to load from GitHub... Check your internet connection.")
