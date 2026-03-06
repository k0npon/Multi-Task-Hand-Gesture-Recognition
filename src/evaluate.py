import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from dataloader import HandGestureDataset
from model import MultiTaskHandNet
from utils import compute_bbox_iou, compute_seg_metrics, get_multitask_loss, set_seed, compute_seg_iou, compute_dice_coefficient
import seaborn as sns
import matplotlib.pyplot as plt


set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskHandNet(num_classes=10).to(device)
model.load_state_dict(torch.load("../weight/weight.pth"))
model.eval()

### VALIDATION
# validation set evaluation
val_dataset = HandGestureDataset(root_dir="/workspace/CW1/RGB_depth_annotations/", split='val')
val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False)

val_preds_class  = []
val_labels_class = []
val_bbox_iou     = 0.0
val_seg_iou      = 0.0
val_dice         = 0.0
val_detections   = 0

print(f"\nEvaluating model on {len(val_dataset)} val samples...")

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        bboxes = batch['bbox'].to(device)
        masks  = batch['mask'].to(device)

        preds = model(images)

        _, predicted = torch.max(preds['class'], 1)
        val_preds_class.extend(predicted.cpu().numpy())
        val_labels_class.extend(labels.cpu().numpy())

        for i in range(images.size(0)):
            iou = compute_bbox_iou(preds['bbox'][i].cpu().numpy(), bboxes[i].cpu().numpy())
            val_bbox_iou += iou
            if iou >= 0.5:
                val_detections += 1
            val_dice    += compute_dice_coefficient(preds['mask'][i], masks[i])
            val_seg_iou += compute_seg_iou(preds['mask'][i], masks[i])

# print validation performance metrics
val_num     = len(val_dataset)
val_acc     = accuracy_score(val_labels_class, val_preds_class) * 100
val_f1      = f1_score(val_labels_class, val_preds_class, average='macro')
val_class_names = list(val_dataset.classes.keys())

print("\n" + "="*30)
print("       FINAL VAL RESULTS      ")
print("="*30)
print(f"Top-1 Accuracy (C):        {val_acc:.2f}%")
print(f"Macro F1-Score (C):        {val_f1:.4f}")
print(f"Detection Acc@0.5 IoU (D): {val_detections/val_num*100:.2f}%")
print(f"Mean BBox IoU (D):         {val_bbox_iou/val_num:.4f}")
print(f"Mean Seg IoU (S):          {val_seg_iou/val_num:.4f}")
print(f"Mean Dice Coefficient (S): {val_dice/val_num:.4f}")

val_conf_mat = confusion_matrix(val_labels_class, val_preds_class)
print("\nVal Confusion Matrix:")
print(val_conf_mat)



# plot validation confusion matrix
val_conf_mat_norm = val_conf_mat.astype(float) / val_conf_mat.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(val_conf_mat_norm, annot=True, fmt='.2f', cmap='Oranges',
            vmin=0, vmax=1, xticklabels=val_class_names, yticklabels=val_class_names, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Val Confusion Matrix (Normalised)\nAccuracy: {val_acc:.2f}%  |  F1: {val_f1:.4f}', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../result/confusion_matrix_val_normalised.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_val_normalised.jpg")


### TEST
# test set evaluation
test_dataset = HandGestureDataset(root_dir="/workspace/CW1/test_data", split='test')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_preds_class = []
all_labels_class = []
total_bbox_iou = 0.0
total_seg_iou = 0.0
total_dice = 0.0
correct_detections = 0 

print(f"Evaluating model on {len(test_dataset)} test samples...")

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        bboxes = batch['bbox'].to(device)
        masks = batch['mask'].to(device)
        
        preds = model(images)
        
        # classification
        _, predicted = torch.max(preds['class'], 1)
        all_preds_class.extend(predicted.cpu().numpy())
        all_labels_class.extend(labels.cpu().numpy())
        
        # detection & segmentation
        for i in range(images.size(0)):
            iou = compute_bbox_iou(preds['bbox'][i].cpu().numpy(), bboxes[i].cpu().numpy())
            total_bbox_iou += iou
            if iou >= 0.5: correct_detections += 1
            
            total_dice += compute_dice_coefficient(preds['mask'][i], masks[i])
            total_seg_iou += compute_seg_iou(preds['mask'][i], masks[i])

# print test performance metrics
num_samples = len(test_dataset)
top1_acc = accuracy_score(all_labels_class, all_preds_class) * 100
f1       = f1_score(all_labels_class, all_preds_class, average='macro')
class_names = list(test_dataset.classes.keys())

print("\n" + "="*30)
print("       FINAL TEST RESULTS      ")
print("="*30)
print(f"Top-1 Accuracy (C):        {top1_acc:.2f}%")
print(f"Macro F1-Score (C):        {f1:.4f}")
print(f"Detection Acc@0.5 IoU (D): {correct_detections/num_samples*100:.2f}%")
print(f"Mean BBox IoU (D):         {total_bbox_iou/num_samples:.4f}")
print(f"Mean Seg IoU (S):          {total_seg_iou/num_samples:.4f}") 
print(f"Mean Dice Coefficient (S): {total_dice/num_samples:.4f}")

conf_mat = confusion_matrix(all_labels_class, all_preds_class)

# print test confusion matrix in terminal
print("\nConfusion Matrix:")
print(conf_mat)

# plot test confusion matrix
conf_mat_norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
            vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix (Normalised)\nAccuracy: {top1_acc:.2f}%  |  F1: {f1:.4f}', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../result/confusion_matrix_normalised.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_normalised.jpg")