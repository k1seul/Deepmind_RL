
    


import torch 
from torch.nn import MSELoss
import numpy as np 
from Env import make_env, ExperienceMemoryBuffer
from collections import deque
import random
from tensorboardX import SummaryWriter
import gc
import gymnasium as gym
import torch.nn as nn 
import torch.nn.functional as F


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



class Agent:
    def __init__(self, writer,env_name = "ALE/Breakout-v5", device='cpu') -> None:
        self.mem = ExperienceMemoryBuffer()
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.network = DQNSimple(n_actions=self.n_actions, device=device)
        self.device = device
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay_rate = (self.epsilon - self.epsilon_final) / 1_000_00
        self.loss = MSELoss() 
        self.optim = torch.optim.RMSprop(self.network.parameters(), lr=0.00025)
        self.gamma = 0.99
        self.total_frames = 10_000_000
        self.writer = writer
        self.td_loss = 0 

        # self.frame_skip = 4

    def reset(self):
        self.state, _ = self.env.reset()
        self.phi_processed_state = deque(maxlen=4)
        for _ in range(4):
            self.phi_processed_state.append(self.state)

    def get_phi_observation(self):
        ## Get recent 4 frams as observation
        obs = np.array(list(self.phi_processed_state), dtype=np.float32)
        # obs = obs.permute(1, 2, 0)
        return obs 


    def play(self, writer):
        t = 0
        episode_num = 0 
        while t < self.total_frames:
            done = False
            truncated = False 
            self.reset()
            total_rewards = 0
            episode_length = 0 

            while not(done or truncated):
                action = self.action()
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_rewards += reward
                
                current_phi = self.get_phi_observation()
                self.state = next_state
                self.phi_processed_state.append(next_state)
                next_phi = self.get_phi_observation()
                # action = torch.tensor(action, device=self.device, dtype=torch.int16)
                # reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
                action = np.array(action)
                reward = np.array(reward)
                done = np.array(done, dtype=np.bool8)
                self.mem.add_experience(current_phi, action, reward, next_phi, done)
                # print(self.mem.sample())

                ## update network
                update_samples = self.mem.sample()
                ## update samples < batch_size
                if update_samples is None:
                    continue 
                self.update(update_samples)
                t += 1 
                episode_length += 1 

            if episode_num % 10 == 0:

                writer.add_scalar('data/rewards', total_rewards, episode_num )
                writer.add_scalar('data/epsilons', self.epsilon, episode_num )
                writer.add_scalar('data/steps', round(t/10_000_000 * 100, 1), episode_num )
                writer.add_scalar('data/episode_length', episode_length, episode_num )
                writer.add_scalar('data/td_error', self.td_loss, episode_num )
                writer.add_scalar('data/performace', self.evaluate_agent(), episode_num)
            print(f"Episode Number: {episode_num}, rewards: {total_rewards}, epsilon: {agent.epsilon}")
            episode_num += 1 

    def evaluate_agent(self):
        eval_episode = 10
        sum_rewards = 0 
        max_frame = 1000 
        for _ in range(eval_episode):
            done = False
            truncated = False 
            self.reset()
            total_rewards = 0
            frame = 0 
            while not(done or truncated) and frame < max_frame:
                frame += 1
                action = self.action(exploit=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_rewards += reward
                
                self.state = next_state
                self.phi_processed_state.append(next_state)
            sum_rewards += total_rewards
        return sum_rewards / 10 

        



    def action(self, exploit = False):
        if random.random() < self.epsilon and not exploit:
            action = random.randint(0, self.n_actions-1)
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_final)
        else:    
            obs = torch.tensor(self.get_phi_observation(), dtype=torch.float32).to(self.device)
            q_values = self.network(obs.reshape(1, *obs.shape))
            action = np.argmax(q_values.detach().cpu().numpy())

        if not exploit:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_final)
        return action
    
    def update(self, samples):
        # print(samples)
        # import pdb; pdb.set_trace()
        states = np.stack(samples[:, 0])
        actions = np.stack(samples[:, 1])
        rewards = np.stack(samples[:, 2])
        next_states = np.stack(samples[:, 3])
        dones = np.stack(samples[:, 4])

        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.int16)

        Q_next_states = self.network(next_states)
        Q_max_next_states = torch.max(Q_next_states, axis=1)[0] 
        y_i = rewards + self.gamma * Q_max_next_states * (1 - dones)
        Q_current_states = self.network(states)
        Q_current_states = Q_current_states[torch.arange(Q_current_states.size(0)), actions]


        loss = self.loss(Q_current_states, y_i)
        # print(loss)
        # import pdb; pdb.set_trace()
        self.td_loss = float(loss.detach().cpu().numpy() / 32)
        self.optim.zero_grad()
        loss.backward() 
        self.optim.step()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    env_names = ["CartPole-v1"
                 ]

    for env in env_names:
        gc.collect()
        writer = SummaryWriter() 
        agent = Agent(env_name=env,writer=writer, device='cuda') 
        agent.play(writer)
        writer.close() 
        