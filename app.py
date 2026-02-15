import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(url):
    """Fetches image from GitHub with verification."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Check if we actually got an image and not a text/html error page
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            st.error(f"URL returned non-image data ({content_type}). Check your path.")
            return None
            
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """Dice overlap math."""
    student_mask = np.where(canvas_data[:,:,3] > 0, 1, 0)
    gt_binary = np.where(np.array(gt_mask.convert("L")) > 0, 1, 0)
    intersection = np.logical_and(student_mask, gt_binary).sum()
    return (2. * intersection) / (student_mask.sum() + gt_binary.sum() + 1e-7)

# --- 2. App Logic ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# VERIFIED RAW URLS from mateuszbuda repository
IMG_URL = "https://raw.githubusercontent.com"
MSK_URL = "https://raw.githubusercontent.com"

bg_image = load_from_repo(IMG_URL)
gt_mask = load_from_repo(MSK_URL)

if bg_image and gt_mask:
    st.subheader("Outline the tumor boundary")
    
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
        choice = st.radio("Pathology Class:", ["LGG", "HGG", "Meningioma"])
        
        if st.button("Submit Result"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            st.divider()
            
            c1, c2 = st.columns(2)
            c1.metric("Dice Score", f"{score:.2%}")
            c2.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
            
            if score > 0.85:
                st.success("Expert Level Match!")
            else:
                st.warning("Needs refinement.")
else:
    st.info("Loading medical images... if this persists, verify the 'raw' URLs in your code.")

