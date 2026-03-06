import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import HandGestureDataset
from model import MultiTaskHandNet
from utils import compute_bbox_iou, compute_seg_metrics, get_multitask_loss, set_seed
import matplotlib.pyplot as plt

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
lr = 0.001

# dataset
train_dataset = HandGestureDataset(root_dir="/workspace/CW1/RGB_depth_annotations/", split='train')
val_dataset = HandGestureDataset(root_dir="/workspace/CW1/RGB_depth_annotations/", split='val')

# dataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# initialize model
model = MultiTaskHandNet(num_classes=10).to(device)

# loss functions 
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()
criterion_mask = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-5)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0
print(f"Starting Training: {len(train_dataset)} train samples, {len(val_dataset)} valid samples")

for epoch in range(epochs):
    #train
    model.train()
    total_train_loss = 0
    train_correct = 0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        bboxes = batch['bbox'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        
        # loss calculation
        loss_c = criterion_class(preds['class'], labels)
        loss_b = criterion_bbox(preds['bbox'], bboxes)
        loss_m = criterion_mask(preds['mask'], masks)
        
        loss = (loss_c * 1.0) + (loss_b * 0.1) + (loss_m * 1.0) 
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # track training accuracy
        _, predicted = torch.max(preds['class'], 1)
        train_correct += (predicted == labels).sum().item()
     
    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = 100. * train_correct / len(train_loader.dataset)
        
    # validation
    model.eval()
    total_val_loss = 0
    val_correct = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)
            masks = batch['mask'].to(device)
            
            preds = model(images)
            
            v_loss_c = criterion_class(preds['class'], labels)
            v_loss_b = criterion_bbox(preds['bbox'], bboxes)
            v_loss_m = criterion_mask(preds['mask'], masks)
            total_val_loss += (v_loss_c + v_loss_b + (v_loss_m * 15.0)).item()
            
            _, predicted = torch.max(preds['class'], 1)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = 100. * val_correct / len(val_loader.dataset) 

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Valid Accuracy: {val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "../weight/weight.pth")
 

# plot loss curve and accuracy curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, epochs+1), train_losses, label='Train Loss')
ax1.plot(range(1, epochs+1), val_losses,   label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
ax2.plot(range(1, epochs+1), val_accs,   label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('../result/training_curves.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("Training Curve Saved!")
