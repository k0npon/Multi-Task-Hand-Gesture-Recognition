# COMP0248 CW1: Multi-Task Hand Gesture Recognition

## Overview
Multi-task deep learning model for simultaneous hand gesture classification, detection, and segmentation from RGB-D data.

## Requirements
pip install -r requirements.txt

## Project Structure
project_25207564_Lui/<br>                                
├── dataset/<br>
│   └── 25207564_Lui/<br>
├── src/<br>
│   ├── dataloader.py<br>
│   ├── model.py<br>
│   ├── train.py<br>
│   ├── evaluate.py<br>
│   ├── visualise.py<br>
│   └── utils.py<br>
├── weight/<br>
│   └── weight.pth<br>
├── result/<br>
│   ├── confusion_matrix_normalised.jpg<br>
│   ├── confusion_matrix_val_normalised.jpg<br>
│   ├── Hand_Gesture_Full_Evaluation.pdf<br>
│   ├── Hand_Gesture_Val_Evaluation.pdf<br>
│   └── training_curves.jpg<br>
├── requirements.txt<br>
└── README.md<br>

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
