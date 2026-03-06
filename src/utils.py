import torch
import numpy as np
import random

def compute_bbox_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def compute_seg_metrics(pred_mask, gt_mask):
    pred = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred * gt_mask).sum()
    total_pixels = pred.sum() + gt_mask.sum()
    dice = (2. * intersection) / (total_pixels + 1e-6)
    iou = intersection / (total_pixels - intersection + 1e-6)
    return iou.item(), dice.item()

def get_multitask_loss(preds, batch, criteria, weights):
    loss_c = criteria['class'](preds['class'], batch['label'])
    loss_b = criteria['bbox'](preds['bbox'], batch['bbox'])
    loss_m = criteria['mask'](preds['mask'], batch['mask'])
    total_loss = (weights['class'] * loss_c + 
                  weights['bbox'] * loss_b + 
                  weights['mask'] * loss_m)
    return total_loss, loss_c, loss_b, loss_m

def compute_dice_coefficient(pred_mask, gt_mask):
    pred = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred * gt_mask).sum()
    return (2. * intersection / (pred.sum() + gt_mask.sum() + 1e-6)).item()

def compute_seg_iou(pred_mask, gt_mask):
    pred = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred * gt_mask).sum()
    union = (pred + gt_mask).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False