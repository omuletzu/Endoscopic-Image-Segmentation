# Endoscopic Image Segmentation with U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv&logoColor=white)

**High-precision semantic segmentation of endoscopic images** using a **U-Net deep learning architecture**, classifying **13 tissue and instrument types**. Achieves **96% pixel accuracy** and **0.1235 test loss** on real clinical images.

This repository implements a **full end-to-end pipeline** from raw image preprocessing to model training, evaluation, and clinically interpretable visualization.

## Tech Stack

- **Languages**: Python 3, C++
- **Deep Learning**: TensorFlow / Keras
- **Image Processing**: OpenCV
- **Data & Utilities**: NumPy, scikit-learn
- **Visualization**: Matplotlib

## File meaninng

- **OpenCVApplication.cpp**: Full processing, augmentation, and image enhancing pipeline.
- **model.ipynb**: U-Net architecture, reading images, training, testing, evaluating, visualizing, and generating plots.
- **main.py**: Inference on new images, generating per-pixel class maps.
- **generate-color-corresponce.ipynb**: Generates class-to-color mapping for segmented images.
- **requirements.txt**: Python dependencies

## Workflow Overview

1. **Image Preprocessing (C++ / OpenCV)**  
   - All images are preprocessed in `OpenCVApplication.cpp` before model inference:  
     - **Resizing** to 256×256  
     - **Grayscale conversion**  
     - **Data augmentation** (rotation, flipping, zoom, brightness/contrast/saturation adjustment)  
     - **Mask preparation** for model training and evaluation  

2. **Model Inference (Python / TensorFlow-Keras)**  
   - `main.py` takes preprocessed images and runs the **trained U-Net model**.  
   - Generates a **class ID map** where each pixel is labeled with its predicted class.

3. **Class-to-Color Mapping (C++ / OpenCV & Python)**  
   - Class ID maps are converted into **final segmented images** using the color correspondence mapping (`generate-color-corresponce.ipynb` + C++ pipelines).  
   - Produces **clinically interpretable, color-coded segmented images** for analysis.

## U-Net Architecture

<img width="1555" height="1036" alt="image" src="https://github.com/user-attachments/assets/a3ef11f6-66e0-44c3-a983-1e3137d308cb" />

## Class - color mapping

<img width="252" height="315" alt="image" src="https://github.com/user-attachments/assets/48d9bdd1-21cd-4288-8443-b561043f9c4e" />

## Example of infered images

<p float="left">
  <img src="https://github.com/user-attachments/assets/030549ab-76ea-4061-9cd0-d03ac2bff5c2" width="350" />
  <img src="https://github.com/user-attachments/assets/4b63c532-c17f-4785-aa2e-e5d94363b620" width="350" />
</p>

<p float="left">
   <img src="https://github.com/user-attachments/assets/8d06e00b-47a1-4b51-921a-2906d4190d20" width="350" />
   <img src="https://github.com/user-attachments/assets/152f1d0e-7135-4c86-919f-9aa84c0567d8" width="350" />
</p>


