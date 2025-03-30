# model.py
import torch.nn as nn
from torchvision import models

class SharkClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super(SharkClassifier, self).__init__()
        # Initialize ResNet50 model
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
