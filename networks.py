import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import math 
from abc import ABC 

class DQN(nn.Module):
    def __init__(self, args, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(args.frame_stack, 16, 8, 4, device=args.device)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, device=args.device)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, device=args.device)
        if args.noisy_net:
            self.fc1 = NoisyLayer(7*7*64, 512, device=args.device)
            self.fc2 = NoisyLayer(512, n_actions, device=args.device)
        else:
            self.fc1 = nn.Linear(7*7*64, 512, device=args.device)
            self.fc2 = nn.Linear(512, n_actions, device=args.device)
        self.device = args.device

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain=math.sqrt(2)) 
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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
    
    # def reset(self):
    #     self.fc1.reset_noise()
    #     self.fc2.reset_noise()

    # def set_zero_noise(self):
    #     self.fc1.set_noise_zero()
    #     self.fc2.set_noise_zero()
    
class DuelDQN(nn.Module):
    def __init__(self, args, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(args.frame_stack, 16, 8, 4, device=args.device)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, device=args.device)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, device=args.device)
        self.scale_grad = ScaleGrad(math.sqrt(2))
        if args.noisy_net:
            self.adv_fc1 = NoisyLayer(7*7*64, 512, device=args.device)
            self.adv_fc2 = NoisyLayer(512, n_actions, device=args.device)
            self.val_fc1 = NoisyLayer(7*7*64, 512,  device=args.device)
            self.val_fc2 = NoisyLayer(512, 1, device=args.device)
        else:
            self.adv_fc1 = nn.Linear(7*7*64, 512, device=args.device)
            self.adv_fc2 = nn.Linear(512, n_actions, device=args.device)
            self.val_fc1 = nn.Linear(7*7*64, 512,  device=args.device)
            self.val_fc2 = nn.Linear(512, 1, device=args.device)
        self.device = args.device

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain=math.sqrt(2)) 
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x) -> torch.tensor:
        x = x /255.0
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) 
        x = x.flatten(start_dim=1)
        x = self.scale_grad(x) 
        adv = self.adv_fc1(x) 
        adv = F.relu(adv) 
        adv = self.adv_fc2(adv)
        val = self.val_fc1(x)
        val = F.relu(val)
        val = self.val_fc2(val)

        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values
    
    # def reset(self):
    #     self.adv_fc1.reset_noise()
    #     self.adv_fc2.reset_noise()
    #     self.val_fc1.reset_noise()
    #     self.val_fc2.reset_noise()

    # def set_zero_noise(self):
    #     self.adv_fc1.set_noise_zero()
    #     self.adv_fc2.set_noise_zero()
    #     self.val_fc1.set_noise_zero()
    #     self.val_fc2.set_noise_zero()

        
class ScaleGrad(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x
    
    def backward(self, grad):
        grad / self.scale
        return grad
    


class DQNSimple(nn.Module):
    """
    2 fully connected layers with
        * relu layers
    """
    def __init__(self, n_actions=2, device='cuda', FRAME_STACK = 4):
        super().__init__()
        self.fc1 = nn.Linear(4 * FRAME_STACK, 64, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(64, 64, device=device, dtype=torch.float32)
        self.fc3 = nn.Linear(64, n_actions, device=device, dtype=torch.float32)


    def forward(self, x) -> torch.tensor:
        x = x.flatten(start_dim=1)
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x 

class DualDQNSimple(nn.Module):
    def __init__(self, n_actions=2, device='cuda', FRAME_STACK = 4):
        """
        2 fully connected layers with different 2nd layer
            * fc1 (relu)
            * fc2 (relu)
                * fc_adv
                * fc_val
        """
        super().__init__()
        self.fc1 = nn.Linear(4 * FRAME_STACK, 64, device=device, dtype=torch.float32)
        self.fc2 = nn.Linear(64, 64, device=device, dtype=torch.float32)
        self.fc_adv = nn.Linear(64, n_actions, device=device, dtype=torch.float32)
        self.fc_val = nn.Linear(64, 1, device=device, dtype=torch.float32)
        self.device = device

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x) 
        
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        
        # Combining value and advantage as per dueling DQN
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values

# class NoisyLayer(nn.Module):
#     def __init__(self, n_inputs, n_outputs, device):
#         super().__init__()
#         self.n_inputs = n_inputs 
#         self.n_outputs = n_outputs 
#         self.device = device 

#         self.mu_w = nn.Parameter(torch.Tensor(self. n_outputs, self.n_inputs)).to(device)
#         self.sigma_w = nn.Parameter(torch.Tensor(self.n_outputs, self.n_inputs)).to(device)
#         self.register_buffer('weight_epsilon', torch.FloatTensor(self.n_outputs, self.n_inputs).to(device))

#         self.mu_b = nn.Parameter(torch.Tensor(self.n_outputs)).to(device)
#         self.sigma_b = nn.Parameter(torch.Tensor(self.n_outputs)).to(device)
#         self.register_buffer('bias_epsilon', torch.FloatTensor(self.n_outputs).to(device))


#         with torch.no_grad():

#             self.mu_w.data.uniform_(-1/math.sqrt(self.n_inputs), 1 / math.sqrt(self.n_inputs))
#             self.sigma_w.fill_(0.1/math.sqrt(self.n_inputs))

#             self.mu_b.data.uniform_(-1/math.sqrt(self.n_inputs), 1/math.sqrt(self.n_inputs))
#             self.sigma_b.data.fill_(0.1/math.sqrt(self.n_outputs))
#         self.reset_noise()


#     def forward(self, inputs):
#         x = inputs
#         weights = self.mu_w + self.sigma_w * self.weight_epsilon
#         biases = self.mu_b + self.sigma_b * self.bias_epsilon
#         x = F.linear(x, weights, biases)
#         return x 
    
#     @staticmethod 
#     def f(x):
#         return torch.sign(x) * torch.sqrt(torch.abs(x))
    
#     def reset_noise(self):
#         epsilon_i = self.f(torch.randn(self.n_inputs, device=self.device))
#         epsilon_j = self.f(torch.randn(self.n_outputs, device=self.device))
#         self.weight_epsilon.copy_(epsilon_j.ger(epsilon_i))
#         self.bias_epsilon.copy_(epsilon_j)

#     def set_noise_zero(self):
#         self.weight_epsilon.copy_(torch.zeros_like(self.weight_epsilon))
#         self.bias_epsilon.copy_(torch.zeros_like(self.bias_epsilon)) 

class NoisyLayer(nn.Linear):
    def __init__(self, n_inputs, n_outputs, device='cuda', sigma_zero=0.4, bias=True):
        super().__init__(n_inputs, n_outputs, bias=bias, device=device)
        sigma_init = sigma_zero / math.sqrt(n_inputs)
        self.sigma_weight = nn.Parameter(torch.full((n_outputs, n_inputs), sigma_init)).to(device)
        self.register_buffer('epsilon_input', torch.zeros(1, n_inputs).to(device))
        self.register_buffer('epsilon_output', torch.zeros(n_outputs, 1).to(device))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((n_outputs,), sigma_init)).to(device)

    def forward(self, x):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda y: torch.sign(y) * torch.sqrt(torch.abs(y))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias 
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t() 
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(x, self.weight + self.sigma_weight * noise_v, bias)