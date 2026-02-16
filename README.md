# MedSigLIP Neuro-Tutor

An interactive medical education application built for the MedGemma Impact Challenge. This tool leverages MedSigLIP (Sigmoid Loss for Language Image Pre-training) to provide medical students with a zero-shot AI interface for identifying brain tumors in MRI scans.

## Project Overview
The MedSigLIP Neuro-Tutor bridges the gap between theoretical neuroanatomy and clinical radiology. Students can interactively segment suspected pathology on T2-FLAIR MRI slices, then verify their findings using a simulated MedSigLIP zero-shot classification pipeline.

## Features
- Interactive Canvas: High-resolution (448x448) drawing interface for manual tumor localization.
- Zero-Shot AI Review: Diagnostic verification based on MedSigLIP vision and text embeddings.
- Clinical Cases: Multi-class training cases including Glioma, Meningioma, and Pituitary tumors.

## Requirements and Libraries
- Streamlit (v1.40.0) - Web application framework.
- Streamlit-Drawable-Canvas - Interactive drawing tools for medical imaging.
- NumPy - Array processing for image data.
- Pillow (PIL) - Image handling and resolution standardization.

## Dataset
This application utilizes standardized MRI samples sourced from Zachary Nguyen's Brain Tumor Classification repository, converted for educational use in 2D format.

## Installation and Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run app.py

## Impact Statement
Submitted for the MedGemma Impact Challenge. This project demonstrates how Health AI Developer Foundations can be used to create human-centered tools that improve medical education and clinical reasoning.

