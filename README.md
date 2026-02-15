This project requires the following libraries:
Streamlit - For the web application framework.
NumPy - For handling image arrays and mask calculations.
Streamlit-Drawable-Canvas - To enable the drawing and polygon tools on medical images.
Dataset Source: LGG MRI Segmentation (Kaggle).
Image Processing: Images must be converted from .tif (standard in the Buda repo) to a web-friendly format or handled via PIL.
Interactivity: Uses streamlit-drawable-canvas to capture student-generated polygons as binary masks
