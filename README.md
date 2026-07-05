# Bear Classification Project

This project implements a machine learning model designed to classify different types of bears using the MobileNetv2 architecture.

## Project Objective
The goal of this project is to develop a robust classification model capable of accurately identifying bear species from input images, facilitating easier wildlife classification and analysis.

## Dataset Overview
The model was trained on a dataset of **309 images** collected from Kaggle. The images are in `.jpg` format and are categorized into the following five classes:

| Bear Type | Number of Images |
| :--- | :--- |
| **Polar Bear** | 100 |
| **Black Bear** | 68 |
| **Teddy Bear** | 50 |
| **Grizzly Bear** | 46 |
| **Panda Bear** | 45 |

## Data Workflow
To ensure compatibility with the MobileNetv2 model, the data underwent the following preprocessing pipeline:

1. **Collect Data:** Sourced from Kaggle.
2. **Resizing:** All images were resized to **(224, 224)** pixels.
3. **Standardization:** Pixel values were normalized by dividing by 255.
4. **Data Splitting:**
   - **Training Set:** 90%
   - **Test Set:** 10%
