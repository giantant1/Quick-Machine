import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(url):
    """
    Fetches the image directly from the GitHub 'raw' endpoint.
    Includes headers to bypass security blocks.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Check that we received an actual image and not an HTML error page
        if 'image' not in response.headers.get('Content-Type', ''):
            st.error("Connection failed: GitHub returned a webpage instead of an image.")
            return None
            
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """
    Calculates Dice Similarity Coefficient (DSC).
    Dice = (2 * Intersection) / (Total Area)
    """
    # Student mask: 1 where drawn (alpha > 0), else 0
    student_mask = (canvas_data[:, :, 3] > 0).astype(np.uint8)
    
    # Expert mask: 1 where pixel > 0, else 0
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = (gt_array > 0).astype(np.uint8)
    
    intersection = np.sum(student_mask * gt_binary)
    dice = (2. * intersection) / (np.sum(student_mask) + np.sum(gt_binary) + 1e-7)
    return dice

# --- 2. App Interface ---

st.set_page_config(page_title="Tumor Quiz", layout="centered")
st.title("Quick Machine: Tumor Segmentation Quiz")
st.markdown("### BraTS Standard: Manual Segmentation")
st.info("Use the **Polygon Tool** on the toolbar to outline the tumor core.")

# Direct Raw Links to Mateusz Buda Repository Assets
IMG_URL = "https://raw.githubusercontent.com"
MSK_URL = "https://raw.githubusercontent.com"

# Load Images
bg_image = load_from_repo(IMG_URL)
gt_mask = load_from_repo(MSK_URL)

if bg_image and gt_mask:
    # Quiz Stage 1: The Canvas
    st.subheader("1. Outline the tumor boundary")
    
    if st.button("Reset Canvas"):
        st.rerun()

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Translucent orange
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key="canvas",
        height=256,
        width=256,
        display_toolbar=True
    )

    # Quiz Stage 2: Submit and Score
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.divider()
        st.subheader("2. Final Diagnosis")
        choice = st.radio("Based on the image, what is the suspected grade?", 
                          ["LGG (Lower-Grade)", "HGG (High-Grade)", "Meningioma"])
        
        if st.button("Submit My Answer"):
            score = calculate_dice(canvas_result.image_data, gt_mask)
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Dice Similarity Score", f"{score:.2%}")
                st.write("**Expert Level:** > 0.85")
                if score > 0.85:
                    st.success(f"Expert Match! Your {choice} diagnosis is correct.")
                else:
                    st.warning("Needs refinement. Check the expert mask.")
            
            with col2:
                st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
else:
    st.warning("🔄 Fetching patient records from the medical database... If this fails, refresh your browser.")


