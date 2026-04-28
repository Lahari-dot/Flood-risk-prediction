import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
