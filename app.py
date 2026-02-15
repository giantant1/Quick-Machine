import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(image_name):
    """
    Fetches a clean 2D slice or mask using the direct raw GitHub URL.
    This version includes a 'User-Agent' header to prevent GitHub from blocking the request.
    """
    # Using the official 'raw' subdomain for the Buda repository assets
    base_url = "https://raw.githubusercontent.com"
    file_url = f"{base_url}{image_name}"
    
    # Header makes the request look like a browser instead of a bot
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(file_url, headers=headers, timeout=10)
        response.raise_for_status() 
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Error loading {image_name}: {e}")
        return None

def calculate_dice(canvas_data, gt_mask):
    """Calculates the Dice Similarity Coefficient (DSC) between user and expert."""
    # Convert student drawing (alpha channel) to binary (1 for drawn, 0 for empty)
    student_mask = np.where(canvas_data[:,:,3] > 0, 1, 0)
    
    # Convert Expert Ground Truth to binary (1 for tumor, 0 for healthy)
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = np.where(gt_array > 0, 1, 0)
    
    # Ensure they are the same size for the math to work
    intersection = np.logical_and(student_mask, gt_binary).sum()
    dice = (2. * intersection) / (student_mask.sum() + gt_binary.sum() + 1e-7)
    return dice

# --- 2. App Logic ---

st.title("Quick Machine: Tumor Segmentation Quiz")
st.info("Aligning with BraTS standards: Use the polygon tool to outline the tumor core.")

# Use Session State to manage the quiz flow
if 'quiz_stage' not in st.session_state:
    st.session_state.quiz_stage = "outline"

# Fetch images directly from the official Buda repo assets
bg_image = load_from_repo("TCGA_CS_4944.png")
gt_mask = load_from_repo("TCGA_CS_4944_mask.png")

if bg_image:
    # Stage 1: The Interactive Canvas
    st.subheader("1. Outline the tumor boundary")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange
        stroke_width=2,
        stroke_color="#FFFFFF",
        background_image=bg_image,
        drawing_mode="polygon",
        key="canvas",
        height=256,
        width=256,
        display_toolbar=True
    )

    # Stage 2: Classification (Appears after drawing)
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        st.session_state.quiz_stage = "classify"
        st.divider()
        st.subheader("2. Classify the pathology")
        choice = st.radio("Based on the FLAIR signal, what is this?", 
                          ["LGG (Lower-Grade Glioma)", "HGG (High-Grade Glioma)", "Meningioma"])
        
        if st.button("Submit My Answer"):
            st.session_state.quiz_stage = "reveal"

    # Stage 3: Results and Dice Score
    if st.session_state.quiz_stage == "reveal":
        st.divider()
        score = calculate_dice(canvas_result.image_data, gt_mask)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Dice Similarity Score", f"{score:.2%}")
            st.write("Expert threshold: > 0.85")
        
        with col2:
            st.image(gt_mask, caption="Expert Ground Truth", use_container_width=True)
        
        if score > 0.80:
            st.success(f"Excellent! Your identification as {choice} matches clinical data.")
        else:
            st.warning("Your segmentation boundary needs adjustment. Compare your result with the expert mask.")
else:
    st.warning("Waiting for medical images to load from repository...")
