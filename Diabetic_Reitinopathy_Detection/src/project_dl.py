


import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split




dataset_dir = '/home/shihas/Documents/diabetic_retinopathy/data'
if not os.path.exists(dataset_dir):
    print(f"Dataset {dataset_dir} not found!")

class0_path = ['03 Moderate NPDR', '04 Severe NPDR', '05 PDR', 
               '06 Mild NPDR, with DME', '07 Moderate NPDR, with DME',
               '08 Severe NPDR, with DME', '09 PDR, with DME']
class1_path = ['01 No DR', '02 Mild NPDR']


# In[16]:


# standard transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation for class 0 
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DRDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_factor=5):
        self.root_dir = root_dir
        self.transform = transform
        self.augment_factor = augment_factor
        self.image_paths = []
        self.labels = []
        
        class0_images = []  # Store images for class 0 separately

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            label = 1 if folder in class1_path else 0 if folder in class0_path else None
            if label is None:
                continue  # Ignore unrecognized folders

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if label == 0:
                        class0_images.append(img_path)
                    else:
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        # Augment Class 0 images
        for img_path in class0_images:
            for _ in range(self.augment_factor):  # Oversample Class 0
                self.image_paths.append(img_path)
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if label == 0 and random.random() < 0.8:
            image = augment_transform(image)  # Applying augmentation randomly
        elif self.transform:
            image = self.transform(image)

        return image, label



dataset = DRDataset(dataset_dir, transform=transform, augment_factor=5)


print(f"Total Images: {len(dataset)}")
class_counts = np.bincount(np.array(dataset.labels))
print(f"Class Counts: {class_counts}") 


# In[17]:



indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, stratify=dataset.labels, random_state=42
)

# Creating PyTorch Subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# DataLoaders for Training & Testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Training Set: {len(train_dataset)} images")
print(f"Testing Set: {len(test_dataset)} images")

