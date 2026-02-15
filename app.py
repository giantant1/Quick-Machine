import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# --- 1. Helper Functions ---

def load_from_repo(image_name, is_mask=False):
    """Fetches a clean 2D slice or mask from the Buda/LGG GitHub repo."""
    base_url = "https://raw.githubusercontent.com"
    # Using a known reliable asset from the repo
    file_url = f"{base_url}TCGA_CS_4944.png" if not is_mask else f"{base_url}TCGA_CS_4944_mask.png"
    
    response = requests.get(file_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        st.error("Failed to load image from repository.")
        return None

def calculate_dice(canvas_data, gt_mask):
    """Calculates Dice score between student canvas and ground truth."""
    # Convert student drawing (alpha channel) to binary 1/0
    student_mask = np.where(canvas_data[:,:,3] > 0, 1, 0)
    
    # Convert Ground Truth to binary 1/0
    gt_array = np.array(gt_mask.convert("L"))
    gt_binary = np.where(gt_array > 0, 1, 0)
    
    intersection = np.logical_and(student_mask, gt_binary).sum()
    dice = (2. * intersection) / (student_mask.sum() + gt_binary.sum() + 1e-7)
    return dice

# --- 2. App Layout ---

st.title("Quick Machine: Tumor Segmentation Quiz")
st.write("Aligning with BraTS standards: Use the polygon tool to outline the tumor core.")

if 'quiz_stage' not in st.session_state:
    st.session_state.quiz_stage = "outline"

# Load data once
bg_image = load_from_repo("TCGA_CS_4944.png")
gt_mask = load_from_repo("TCGA_CS_4944_mask.png", is_mask=True)

# Stage 1: The Canvas
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

# Stage 2: Multiple Choice & Submit
if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
    st.session_state.quiz_stage = "classify"
    st.divider()
    st.subheader("2. Classify the pathology")
    choice = st.radio("Based on the FLAIR signal, what is this?", ["LGG (Lower-Grade Glioma)", "HGG (High-Grade Glioma)", "Meningioma"])
    
    if st.button("Submit My Answer"):
        st.session_state.quiz_stage = "reveal"

# Stage 3: The Reveal
if st.session_state.quiz_stage == "reveal":
    st.divider()
    score = calculate_dice(canvas_result.image_data, gt_mask)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Dice Similarity Score", f"{score:.2%}")
        st.write("A score > 0.85 is considered expert level.")
    
    with col2:
        st.image(gt_mask, caption="Radiologist Ground Truth", use_container_width=True)
    
    if score > 0.80:
        st.success(f"Great job! Your identification as {choice} matches the clinical data.")
    else:
        st.warning("Your segmentation missed some boundary areas. Compare your outline to the ground truth.")
