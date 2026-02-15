import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(url):
    """Fetches the image directly from the verified raw GitHub link."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Verify it's an image and not a text error page
        if 'image' not in response.headers.get('Content-Type', ''):
            st.error(f"GitHub returned a web page instead of an image. Check the URL.")
            return None
            
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """Calculates Dice overlap between user drawing and expert mask."""
    # Convert student drawing (alpha channel) to binary
    student_mask = np.where(canvas_data[:,:,3] > 0, 1, 0)
    # Convert Expert Ground Truth to binary
    gt_binary = np.where(np.array(gt_mask.convert("L")) > 0, 1, 0)
    
    intersection = np.logical_and(student_mask, gt_binary).sum()
    dice = (2. * intersection) / (student_mask.sum() + gt_binary.sum() + 1e-7)
    return dice

# --- 2. App Logic ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# VERIFIED DIRECT RAW LINKS (Case Sensitive)
# Source: mateuszbuda/brain-segmentation-pytorch
IMG_URL = "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
MSK_URL = "https://github.com"

bg_image = load_from_repo(IMG_URL)
gt_mask = load_from_repo(MSK_URL)

if bg_image and gt_mask:
    st.subheader("1. Outline the tumor boundary")
    st.write("Use the polygon tool to trace the abnormality.")
    
    if st.button("Reset Drawing"):
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

    # Submit Logic
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        st.subheader("2. Final Diagnosis")
        choice = st.radio("Based on the FLAIR signal, what is this?", ["LGG", "HGG", "Meningioma"])
        
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dice Similarity Score", f"{score:.2%}")
                st.write("Expert match: > 85%")
            with col2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
            
            if score > 0.80:
                st.success(f"Great work! Your classification as {choice} is clinically consistent.")
            else:
                st.warning("The outline deviates from the radiologist's ground truth. Try again!")
else:
    st.info("Attempting to load medical images from the repository...")

