from .base import BaseAgent
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm 
import copy
from src.common.bypass_bn import enable_running_stats, disable_running_stats

class DDQN(BaseAgent):
    name = 'ddqn'
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__(cfg, device, train_env, eval_env, logger, buffer, aug_func, model)  
        self.target_model = copy.deepcopy(self.model).to(self.device)   
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def predict(self, model, obs, eps, exp_noise) -> torch.Tensor:
        q_value, _ = model(obs)
        argmax_action = torch.argmax(q_value, 1).item()
        action = argmax_action

        p = random.random()
        if p < eps:
            action = random.randint(0, self.cfg.action_size-1)
        else:
            action = argmax_action  
        
        return action
    
    def forward(self, online_model, target_model, batch, mode, reduction='mean', reset_noise=True):

        # get samples from buffer
        idxs = batch['idxs']
        obs_batch = batch['obs']
        act_batch = batch['act']
        return_batch = batch['return']
        done_batch = batch['done']
        next_obs_batch = batch['next_obs']
        weights = batch['weights']

        # Calculate current state's q-value 
        cur_q, _ = online_model(obs_batch)
        act_idx = act_batch.reshape(-1,1)
        pred_q = cur_q.gather(1, act_idx).squeeze(1)

        with torch.no_grad():
            next_online_q, _ = online_model(next_obs_batch)
            next_target_q, _ = target_model(next_obs_batch)
            next_act = torch.argmax(next_online_q, 1).reshape(-1,1)
            next_target_q = next_target_q.gather(1, next_act).squeeze(1)
            gamma = (self.cfg.gamma ** self.buffer.n_step)
            target_q = return_batch + (1 - done_batch) * gamma * next_target_q 

      
        square_error = torch.square(target_q - pred_q)
        abs_error = torch.abs(target_q - pred_q)
        if reduction == 'mean':
            loss = (square_error * weights).mean()
        else:
            loss = (square_error * weights)

        # update priority
        if (self.buffer.name == 'per_buffer') and (mode == 'train'):
            self.buffer.update_priorities(idxs=idxs, priorities=abs_error.detach().cpu().numpy())

        # prediction and target
        preds = pred_q 
        targets = target_q

        return loss, preds, targets