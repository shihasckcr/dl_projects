import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_model(pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights= None

    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features             # gets the input features of last layer

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512,2)
    )

    return model

if __name__ == "__main__":
    model = get_resnet_model()
    print(model)

