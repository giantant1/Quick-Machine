import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(url):
    """
    Fetches image data from the raw GitHub endpoint.
    Includes headers to bypass bot-blocking security.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Security check: Ensure we didn't get an HTML error page
        if 'image' not in response.headers.get('Content-Type', ''):
            st.error("GitHub provided a webpage instead of an image. Verify the 'raw' URL.")
            return None
            
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """
    Calculates Dice Similarity Coefficient (DSC).
    Formula: (2 * intersection) / (total pixels in both masks)
    """
    # Student mask: 1 where drawn (alpha > 0), else 0
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    
    # Expert mask: 1 where pixel > 0, else 0
    gt_binary = (np.array(gt_mask.convert("L")) > 0).astype(np.uint8)
    
    intersection = np.sum(student_mask * gt_binary)
    dice = (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)
    return dice

# --- 2. App UI ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")

# OFFICIAL RAW LINKS - These bypass the HTML preview page
IMG_URL = "https://raw.githubusercontent.com"
MSK_URL = "https://raw.githubusercontent.com"

bg_image = load_from_repo(IMG_URL)
gt_mask = load_from_repo(MSK_URL)

if bg_image and gt_mask:
    st.subheader("1. Outline the tumor boundary")
    st.caption("Select the 'Polygon' tool from the toolbar and click to create points.")
    
    if st.button("Reset Canvas"):
        st.rerun()

    # The Canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Translucent orange
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key="canvas",
        height=256,
        width=256,
    )

    # Progression Logic
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        st.subheader("2. Classify the pathology")
        choice = st.radio("What is the suspected grade?", ["LGG", "HGG", "Meningioma"])
        
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Dice Score", f"{score:.2%}")
                st.write("Expert threshold: > 0.85")
                if score > 0.80:
                    st.success(f"Great match for {choice}!")
                else:
                    st.warning("Try refining your outline.")
            with c2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
else:
    st.info("🔄 Connecting to the medical database... If this takes more than 10 seconds, refresh the page.")

