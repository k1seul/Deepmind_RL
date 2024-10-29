import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_actions=4, device='cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4, device=device, dtype=torch.float32)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, device=device, dtype=torch.float32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, device=device, dtype=torch.float32)
        self.fc1 = nn.Linear(7*7*64, 512, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(512, n_actions, device=device, dtype=torch.float32)

    def forward(self, x) -> torch.tensor:
        x = x /255.0
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) 
        x = x.flatten(start_dim=1)
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.fc2(x)
        return x 
    

class DQNSimple(nn.Module):
    def __init__(self, n_actions=2, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(16, 256, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(256, n_actions, device=device, dtype=torch.float32)

    def forward(self, x) -> torch.tensor:
        x = x.flatten(start_dim=1)
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.fc2(x)
        return x 

    
    

        
