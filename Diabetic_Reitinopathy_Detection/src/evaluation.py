import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from project_dl import dataset  
from model import get_resnet_model 
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices = list(range(len(dataset)))
_, test_indices = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=42)
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#loading trained model
model = get_resnet_model(pretrained=False).to(device)
model.load_state_dict(torch.load("resnet_dr_model.pth", map_location=device))  
model.eval()  # Set to evaluation mode

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Non-Referable", "Referable"])

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:  
        images = images.to(device)  
        labels = labels.to(device)
        
        outputs = model(images)  # Get model predictions
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability for Class 1 (Referable)

        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()