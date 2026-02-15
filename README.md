# Quick Machine: Tumor Segmentation Quiz

An interactive medical imaging application built with Streamlit. This tool allows students to perform manual tumor segmentation and compare their accuracy against expert ground truth masks.

## Requirements and Libraries
This project requires the following libraries:
* Streamlit - Web application framework.
* NumPy - Image array and mask calculations.
* Streamlit-Drawable-Canvas - Interactive polygon tools for medical images.
* Pillow (PIL) - Image processing and format conversion.
* Requests - Fetching image data from external repositories.

## Data and Methodology
* Dataset Source: LGG MRI Segmentation (Kaggle) via the Buda Repository.
* Image Processing: Standard .tif files from the Buda repository are handled via PIL and converted to web-friendly formats for the user interface.
* Interactivity: Student-generated polygons are captured via streamlit-drawable-canvas and converted into binary masks to calculate the Dice Similarity Coefficient (DSC).

## Installation and Setup
1. Install dependencies:
   pip install streamlit numpy streamlit-drawable-canvas Pillow requests

2. Run the application:
   streamlit run app.py

