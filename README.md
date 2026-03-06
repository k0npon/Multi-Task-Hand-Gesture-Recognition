# COMP0248 CW1: Multi-Task Hand Gesture Recognition

## Overview
Multi-task deep learning model for simultaneous hand gesture classification, detection, and segmentation from RGB-D data.

## Requirements
pip install -r requirements.txt

## Project Structure
project_25207564_Lui/
├── dataset/
│   └── 25207564_Lui/
├── src/
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualise.py
│   └── utils.py
├── weight/
│   └── weight.pth
├── result/
│   ├── confusion_matrix_normalised.jpg
│   ├── confusion_matrix_val_normalised.jpg
│   ├── Hand_Gesture_Full_Evaluation.pdf
│   ├── Hand_Gesture_Val_Evaluation.pdf
│   └── training_curves.jpg
├── requirements.txt
└── README.md

## Setup
change the path inside HandGestureDataset(root_dir="/workspace/CW1/RGB_depth_annotations/") with the correct directory of training/validation/test set in train.py, evaluate.py and visualise.py

cd src/

## Usage

### Train
python train.py

### Evaluate
python evaluate.py

### Visualise
python visualise.py
