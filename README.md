# RSNA Pneumonia Detection Model

This project is a deep learning-based pneumonia detection system trained on chest X-ray images from the RSNA dataset. It uses a ResNet-50 convolutional neural network to classify whether a patient has pneumonia or not.

# Project Overview

Pneumonia is a serious lung infection that can be detected using chest X-ray images. This project builds a convolutional neural network model to automatically classify X-rays into:

Normal
Pneumonia

The model is trained using PyTorch and transfer learning with a pretrained ResNet-50 architecture.

# Model Architecture
Architecture: ResNet-50
Framework: PyTorch
Type: Binary image classification
Input: Chest X-ray images
Output: Probability of pneumonia
# Dataset

The project uses the RSNA Pneumonia Detection dataset:

Chest X-ray images
Bounding box annotations (optional)
Labels: Normal / Pneumonia

Dataset source: RSNA Pneumonia Detection Challenge




## Project Structure

```text
rsna/
├── model.ipynb                  # Training notebook
├── best_pneumonia_model.pth     # Trained model weights
├── stage_2_train_images/        # Training images (not uploaded to GitHub)
├── stage_2_test_images/         # Test images
├── stage_2_train_labels.csv     # Training labels
├── stage_2_sample_submission.csv
└── README.md
```
# Installation

Install required dependencies:

pip install torch torchvision numpy pandas matplotlib opencv-python

# How to Run
Load the trained model

```
import torch
from torchvision import models

model = models.resnet50()
model.load_state_dict(torch.load("best_pneumonia_model.pth", map_location="cpu"))
model.eval()
```

# Run inference

```
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open("test_xray.jpg")
image = transform(image).unsqueeze(0)

output = model(image)
print(output)
```
# Results
Trained using RSNA chest X-ray dataset
Uses ResNet-50 backbone for feature extraction
Designed for binary classification of pneumonia detection
# Notes
Large dataset folders are not included in this repository due to GitHub size limits
The pretrained model file may need to be stored externally if too large
This project is for educational and research purposes only
# Author

Deep learning project built using PyTorch and the RSNA dataset for medical image classification.