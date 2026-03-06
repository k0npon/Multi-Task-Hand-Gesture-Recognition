import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, img_size=(224, 224), split='train'):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        self.classes = {
            "G01_call": 0, "G02_dislike": 1, "G03_like": 2, "G04_ok": 3,
            "G05_one": 4, "G06_palm": 5, "G07_peace": 6, "G08_rock": 7,
            "G09_stop": 8, "G10_three": 9
        }
        self.samples = []

        if self.split == 'test':
            for gesture_name, label in self.classes.items():
                gesture_dir = os.path.join(root_dir, gesture_name)
                if not os.path.exists(gesture_dir): continue
                
                for clip_folder in os.listdir(gesture_dir):
                    clip_dir = os.path.join(gesture_dir, clip_folder)
                    annotation_dir = os.path.join(clip_dir, "annotation")
                    if not os.path.exists(annotation_dir): continue
                        
                    for mask_name in os.listdir(annotation_dir):
                        if not mask_name.endswith('.png'): continue
                        
                        mask_path = os.path.join(annotation_dir, mask_name)
                        rgb_path = os.path.join(clip_dir, "rgb", mask_name)
                        depth_path = os.path.join(clip_dir, "depth_raw", mask_name.replace('.png', '.npy'))
                        
                        if os.path.exists(rgb_path) and os.path.exists(depth_path):
                            self.samples.append({
                                'rgb': rgb_path, 'depth': depth_path, 
                                'mask': mask_path, 'label': label
                            })

        else:
            all_students = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
            random.seed(42)
            random.shuffle(all_students)

            split_idx = int(0.8 * len(all_students))
            target_students = all_students[:split_idx] if self.split == 'train' else all_students[split_idx:]

            for student in target_students:
                student_path = os.path.join(root_dir, student)
                for gesture_name, label in self.classes.items():
                    gesture_dir = os.path.join(student_path, gesture_name)
                    if not os.path.exists(gesture_dir): continue
                    for clip_folder in os.listdir(gesture_dir):
                        clip_dir = os.path.join(gesture_dir, clip_folder)
                        annotation_dir = os.path.join(clip_dir, "annotation")
                        if not os.path.exists(annotation_dir): continue
                        for mask_name in os.listdir(annotation_dir):
                            if not mask_name.endswith('.png'): continue
                            mask_path = os.path.join(annotation_dir, mask_name)
                            rgb_path = os.path.join(clip_dir, "rgb", mask_name)
                            depth_path = os.path.join(clip_dir, "depth_raw", mask_name.replace('.png', '.npy'))
                            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                                self.samples.append({'rgb': rgb_path, 'depth': depth_path, 'mask': mask_path, 'label': label})
            
            # augmentation for training set
            if self.split == 'train':
                random.seed(42)
                aug_copies = random.sample(self.samples, min(2000, len(self.samples)))
                aug_copies = [dict(s, augment=True) for s in aug_copies]
                self.samples = self.samples + aug_copies

        random.shuffle(self.samples)
        print(f"{self.split.upper()} set: {len(self.samples)} total frames loaded.")

    def __len__(self):
        return len(self.samples)

    def augment(self, image, mask):
        # horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # translation 
        if random.random() < 0.5:
            max_dx = 0.1 * self.img_size[0]
            max_dy = 0.1 * self.img_size[1]
            dx = int(random.uniform(-max_dx, max_dx))
            dy = int(random.uniform(-max_dy, max_dy))
            image = TF.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0)
            mask = TF.affine(mask, angle=0, translate=(dx, dy), scale=1, shear=0,
                            interpolation=TF.InterpolationMode.NEAREST)
        # brightness
        if random.random() < 0.5:
            image[:3] = TF.adjust_brightness(image[:3], random.uniform(0.7, 1.3))
        # contrast 
        if random.random() < 0.5:
            image[:3] = TF.adjust_contrast(image[:3], random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            darkness_factor = random.uniform(0.0, 0.2)  
            image[:3] = image[:3] * darkness_factor  
        return image, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load and resize
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(sample['rgb']), cv2.COLOR_BGR2RGB), self.img_size)
        
        # depth processing 
        depth = np.load(sample['depth'])
        depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)
        depth = np.clip(depth, 400, 1100).astype(np.float32) 
        depth = (depth - 400) / 700.0 
        
        # merge rgbd
        rgbd = np.concatenate((rgb.astype(np.float32) / 255.0, depth[..., None]), axis=-1)
        rgbd = torch.tensor(rgbd).permute(2, 0, 1) 
        
        # load mask 
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0 

        # augment
        if sample.get('augment', False):
            rgbd, mask = self.augment(rgbd, mask)
        
        # bbox extraction 
        coords = torch.where(mask[0] > 0.5)
        if coords[0].numel() > 0:
            bbox = torch.tensor([coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()]).float()
        else:
            bbox = torch.tensor([0., 0., 0., 0.]).float()
        
        return {'image': rgbd, 'mask': mask, 'bbox': bbox, 'label': torch.tensor(sample['label'])}