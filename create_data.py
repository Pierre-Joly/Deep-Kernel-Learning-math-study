import torch
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

class OlivettiDataset(Dataset):
    def __init__(self, num_augmentations=10, crop_ratio=0.7):
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        self.original_images = data.images
        self.num_augmentations = num_augmentations
        self.crop_size = int(64 * crop_ratio)
        self.augmented_images = []
        self.labels = []
        
        self.center_crop = transforms.CenterCrop(self.crop_size)
        self.resize = transforms.Resize((64, 64), interpolation=Image.BICUBIC)
        self.to_tensor = transforms.ToTensor()

        for img in self.original_images:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            for _ in range(num_augmentations):
                # angle en float python
                rotation_angle = random.uniform(-45, 45)
                
                # rotation
                rotated = transforms.functional.rotate(pil_img, rotation_angle, expand=True)
                
                # centre-crop, resize, to_tensor
                cropped = self.center_crop(rotated)
                resized = self.resize(cropped)
                tensor_img = self.to_tensor(resized).float()
                
                # label en tensor float32
                angle_tensor = torch.tensor(rotation_angle, dtype=torch.float32)
                
                self.augmented_images.append(tensor_img)
                self.labels.append(angle_tensor)

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        return self.augmented_images[idx], self.labels[idx]

dataset = OlivettiDataset(num_augmentations=5, crop_ratio=0.7)
torch.save(dataset, 'datasets/olivetti_dataset.pt')
