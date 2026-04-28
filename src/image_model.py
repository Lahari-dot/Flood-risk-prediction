import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageModel, self).__init__()
        # Input shape: (3, 64, 64)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After 2 max pools of 2x2, 64x64 spatial size implies 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 16x32x32
        x = self.pool(F.relu(self.conv2(x))) # 32x16x16
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
