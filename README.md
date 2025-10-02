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
