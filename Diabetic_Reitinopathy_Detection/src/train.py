import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from model import get_resnet_model
import os
from sklearn.model_selection import train_test_split
from project_dl import dataset




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, stratify=dataset.labels, random_state=42
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
MODEL_PATH = "resnet_dr_model.pth"


model = get_resnet_model(pretrained=True).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded existing model from {MODEL_PATH}")

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training Complete! ")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Training Complete! Model saved at {MODEL_PATH}")


