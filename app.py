import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

# 1. Setup Session State
if 'quiz_stage' not in st.session_state:
    st.session_state.quiz_stage = "outline" # Stages: outline -> classify -> reveal

# 2. Stage 1: Outline the Tumor
st.subheader("1. Outline the tumor boundary")
canvas_result = st_canvas(
    background_image=load_from_repo("patient_0017.png"),
    drawing_mode="polygon",
    key="canvas"
)

# 3. Stage 2: Multiple Choice
# Only show if the student has drawn something on the canvas
if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
    st.session_state.quiz_stage = "classify"
    choice = st.radio("What type of tumor is shown?", ["LGG", "HGG", "Meningioma"])
    
    if st.button("Submit My Answer"):
        st.session_state.quiz_stage = "reveal"

# 4. Stage 3: Reveal and Dice Score
if st.session_state.quiz_stage == "reveal":
    # Calculate overlap between student drawing and ground truth mask
    # DSC = (2 * intersection) / (sum_of_pixels)
    score = calculate_dice(canvas_result.image_data, ground_truth_mask)
    st.metric("Dice Similarity Score", f"{score:.2%}")
    st.image(ground_truth_mask, caption="Expert Ground Truth Reveal")
