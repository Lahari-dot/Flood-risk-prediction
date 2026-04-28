import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(TabularFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return F.relu(self.fc1(x))

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class FusionModel(nn.Module):
    def __init__(self, tabular_input_dim, num_classes=2):
        super(FusionModel, self).__init__()
        self.tabular_extractor = TabularFeatureExtractor(input_dim=tabular_input_dim, hidden_dim=16)
        self.image_extractor = ImageFeatureExtractor()
        
        # Combined size = 16 (tabular) + 128 (image) = 144
        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, tab, img):
        tab_features = self.tabular_extractor(tab)
        img_features = self.image_extractor(img)
        
        # Concatenate features
        combined = torch.cat((tab_features, img_features), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
