# Bear Classification Project

This project implements a machine learning model designed to classify different types of bears using the MobileNetv2 architecture. By leveraging transfer learning, the model achieves high accuracy in identifying bear species from input images.

## Project Objective
The goal of this project is to develop a robust classification model capable of accurately identifying bear species, facilitating easier wildlife classification and analysis.

## Dataset Overview
The model was trained on a dataset of **309 images** collected from Kaggle. The dataset is categorized into the following five classes:

| Bear Type | Number of Images |
| :--- | :--- |
| **Polar Bear** | 100 |
| **Black Bear** | 68 |
| **Teddy Bear** | 50 |
| **Grizzly Bear** | 46 |
| **Panda Bear** | 45 |

## Data Workflow
To ensure compatibility with the MobileNetv2 model, the data underwent the following preprocessing pipeline:

1. **Resizing:** All images were resized to **(224, 224)** pixels.
2. **Standardization:** Pixel values were normalized by dividing by 255.
3. **Data Splitting:**
   - **Training Set:** 90%
   - **Test Set:** 10%

## Methodology & Model Training
This project employs a **Transfer Learning** approach, leveraging the pre-trained MobileNetV2 architecture (trained on ImageNet).

### Training Strategy
* **Freeze Base Layer:** The pre-trained layers are frozen to retain learned features from ImageNet.
* **Custom Layers:** A custom set of fully connected (dense) layers was added and trained specifically for this multi-class classification task.
* **Loss Function:** Categorical Cross-Entropy.
* **Optimizer:** Adam optimizer with a learning rate of **0.001**.
