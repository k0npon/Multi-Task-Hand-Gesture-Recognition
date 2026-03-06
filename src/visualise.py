import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from dataloader import HandGestureDataset
from model import MultiTaskHandNet
from utils import compute_bbox_iou, compute_seg_metrics, get_multitask_loss, set_seed
import os

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskHandNet(num_classes=10).to(device)
model.load_state_dict(torch.load("../weight/weight.pth"))
model.eval()

### TEST 
dataset = HandGestureDataset(root_dir="/workspace/CW1/test_data", split='test')
class_names = [k for k in dataset.classes.keys()]

# create test visualization
with PdfPages('../result/Hand_Gesture_Full_Evaluation.pdf') as pdf:
    print("Generating Full Evaluation PDF...")

    num_to_show = min(400, len(dataset))

    for i in range(0, num_to_show, 4):
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        plt.suptitle(f"Test Evaluation: Samples {i} to {i+3}", fontsize=20, y=0.98)

        for j in range(4):
            idx = i + j
            if idx >= len(dataset):
                for ax in axes[j]:
                    ax.axis('off')
                continue

            raw_sample = dataset.samples[idx]   
            data       = dataset[idx]            

            img_tensor = data['image'].unsqueeze(0).to(device)
            gt_label   = data['label'].item()
            gt_mask    = data['mask'][0].numpy()  

            with torch.no_grad():
                preds = model(img_tensor)
        
            _, pred_class_idx = torch.max(preds['class'], 1)
            pred_class_idx    = pred_class_idx.item()
            pred_mask         = torch.sigmoid(preds['mask'][0, 0]).cpu().numpy()
            binary_mask       = (pred_mask > 0.5).astype(np.float32)
            box               = preds['bbox'][0].cpu().numpy().astype(int)
            x1, y1, x2, y2   = np.clip(box, 0, 223)

            rgb       = img_tensor[0, :3, :, :].cpu().permute(1, 2, 0).numpy()
            rgb       = np.clip(rgb, 0, 1)
            depth_map = img_tensor[0, 3, :, :].cpu().numpy()

            rgb_path  = raw_sample['rgb']
            clip_dir  = os.path.dirname(os.path.dirname(rgb_path))
            clip_label = os.path.join(
                os.path.basename(os.path.dirname(clip_dir)),
                os.path.basename(clip_dir)
            )

            correct = pred_class_idx == gt_label
            color   = 'green' if correct else 'red'

            # Original RGB + clip path + GT/Pred 
            axes[j, 0].imshow(rgb)
            axes[j, 0].set_title(
                f"{clip_label}\nGT: {class_names[gt_label]}\nPred: {class_names[pred_class_idx]}",
                color=color, fontsize=9, fontweight='bold'
            )

            # Depth map 
            axes[j, 1].imshow(depth_map, cmap='viridis')
            axes[j, 1].set_title("Depth Channel", fontsize=9)

            # RGB + predicted mask (green transparent) + bbox 
            axes[j, 2].imshow(rgb)
            pred_overlay          = np.zeros((*binary_mask.shape, 4))
            pred_overlay[..., 0]  = 0.0          # R
            pred_overlay[..., 1]  = 1.0          # G
            pred_overlay[..., 2]  = 0.4          # B
            pred_overlay[..., 3]  = binary_mask * 0.5  # Alpha
            axes[j, 2].imshow(pred_overlay)
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, color='cyan', linewidth=2)
            axes[j, 2].add_patch(rect)
            axes[j, 2].set_title("Pred Mask + BBox", fontsize=9)

            # RGB + GT annotation (red transparent)
            axes[j, 3].imshow(rgb)
            gt_overlay          = np.zeros((*gt_mask.shape, 4))
            gt_overlay[..., 0]  = 1.0          # R
            gt_overlay[..., 1]  = 0.2          # G
            gt_overlay[..., 2]  = 0.2          # B
            gt_overlay[..., 3]  = gt_mask * 0.5  # Alpha
            axes[j, 3].imshow(gt_overlay)
            axes[j, 3].set_title("GT Annotation", fontsize=9)

            for ax in axes[j]:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print("Finished! File saved as 'Hand_Gesture_Full_Evaluation.pdf'")



### VALIDATION
val_dataset = HandGestureDataset(root_dir="/workspace/CW1/RGB_depth_annotations/", split='val')
val_class_names = [k for k in val_dataset.classes.keys()]

#create validation visualization
with PdfPages('../result/Hand_Gesture_Val_Evaluation.pdf') as pdf:
    print("Generating Validation Evaluation PDF...")

    num_to_show = min(100, len(val_dataset))

    for i in range(0, num_to_show, 4):
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        plt.suptitle(f"Validation Evaluation: Samples {i} to {i+3}", fontsize=20, y=0.98)

        for j in range(4):
            idx = i + j
            if idx >= len(val_dataset):
                for ax in axes[j]:
                    ax.axis('off')
                continue

            raw_sample = val_dataset.samples[idx]
            data       = val_dataset[idx]

            img_tensor = data['image'].unsqueeze(0).to(device)
            gt_label   = data['label'].item()
            gt_mask    = data['mask'][0].numpy()

            with torch.no_grad():
                preds = model(img_tensor)

            _, pred_class_idx = torch.max(preds['class'], 1)
            pred_class_idx    = pred_class_idx.item()
            pred_mask         = torch.sigmoid(preds['mask'][0, 0]).cpu().numpy()
            binary_mask       = (pred_mask > 0.5).astype(np.float32)
            box               = preds['bbox'][0].cpu().numpy().astype(int)
            x1, y1, x2, y2   = np.clip(box, 0, 223)

            rgb       = img_tensor[0, :3, :, :].cpu().permute(1, 2, 0).numpy()
            rgb       = np.clip(rgb, 0, 1)
            depth_map = img_tensor[0, 3, :, :].cpu().numpy()

            rgb_path   = raw_sample['rgb']
            clip_dir   = os.path.dirname(os.path.dirname(rgb_path))
            clip_label = os.path.join(
                os.path.basename(os.path.dirname(clip_dir)),
                os.path.basename(clip_dir)
            )

            correct = pred_class_idx == gt_label
            color   = 'green' if correct else 'red'

            axes[j, 0].imshow(rgb)
            axes[j, 0].set_title(
                f"{clip_label}\nGT: {val_class_names[gt_label]}\nPred: {val_class_names[pred_class_idx]}",
                color=color, fontsize=9, fontweight='bold'
            )

            axes[j, 1].imshow(depth_map, cmap='viridis')
            axes[j, 1].set_title("Depth Channel", fontsize=9)

            axes[j, 2].imshow(rgb)
            pred_overlay         = np.zeros((*binary_mask.shape, 4))
            pred_overlay[..., 1] = 1.0
            pred_overlay[..., 2] = 0.4
            pred_overlay[..., 3] = binary_mask * 0.5
            axes[j, 2].imshow(pred_overlay)
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, color='cyan', linewidth=2)
            axes[j, 2].add_patch(rect)
            axes[j, 2].set_title("Pred Mask + BBox", fontsize=9)

            axes[j, 3].imshow(rgb)
            gt_overlay         = np.zeros((*gt_mask.shape, 4))
            gt_overlay[..., 0] = 1.0
            gt_overlay[..., 1] = 0.2
            gt_overlay[..., 2] = 0.2
            gt_overlay[..., 3] = gt_mask * 0.5
            axes[j, 3].imshow(gt_overlay)
            axes[j, 3].set_title("GT Annotation", fontsize=9)

            for ax in axes[j]:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print("Finished! File saved as 'Hand_Gesture_Val_Evaluation.pdf'")