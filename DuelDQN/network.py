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
        self.device = device

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
    

class DuelDQN(nn.Module):
    def __init__(self, n_actions=4, device='cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4, device=device, dtype=torch.float32)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, device=device, dtype=torch.float32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, device=device, dtype=torch.float32)
        self.adv_fc1 = nn.Linear(7*7*64, 512, device=device, dtype=torch.float32)
        self.adv_fc2 = nn.Linear(512, n_actions, device=device, dtype=torch.float32)
        self.val_fc1 = nn.Linear(7*7*64, 512,  device=device, dtype=torch.float32)
        self.val_fc2 = nn.Linear(512, 1, device=device, dtype=torch.float32)
        self.device = device

    def forward(self, x) -> torch.tensor:
        x = x /255.0
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) 
        x = x.flatten(start_dim=1)
        adv = self.adv_fc1(x) 
        adv = F.relu(adv) 
        adv = self.adv_fc2(adv)
        val = self.val_fc1(x)
        val = F.relu(val)
        val = self.val_fc2(val)

        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values
    
    def backward(self, loss):
        # Zero gradients
        # loss.zero_grad()
        
        # Backward pass for both streams
        loss.backward()


        # Scale gradients of convolutional layers by 1/sqrt(2)
        scale_factor = 1 / torch.sqrt(torch.tensor(2.0, device=self.device))
        for layer in [self.conv1, self.conv2, self.conv3]:
            if layer.weight.grad is not None:
                layer.weight.grad *= scale_factor
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad *= scale_factor

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
    

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

class DualDQNSimple(nn.Module):
    def __init__(self, n_actions=2, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(16, 256, device=device, dtype=torch.float32)
        self.fc_adv = nn.Linear(256, n_actions, device=device, dtype=torch.float32)
        self.fc_val = nn.Linear(256, 1, device=device, dtype=torch.float32)
        self.device = device

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        
        # Combining value and advantage as per dueling DQN
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values


    
    

        
