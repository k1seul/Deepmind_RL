from Env import make_env, show_img
from network import DQN
from collections import deque
import numpy as np
import torch 


class Agent:
    def __init__(self, env_name = "ALE/Breakout-v5", device='cpu') -> None:
        self.env = make_env(env_name=env_name)
        self.n_actions = self.env.action_space.n
        self.network = DQN(n_actions=self.n_actions, device=device)
        state_dict = torch.load(f"data/{env_name[4:]}.pt")
        self.network.load_state_dict(state_dict) 
        self.device = device

    def reset(self):
        self.state, _ = self.env.reset()
        self.states_mem = deque(maxlen=4)
        for _ in range(4):
            self.states_mem.append(self.state)
        return self.state 

    def play(self):
        t = 0
        done = False
        truncated = False 
        self.reset()
        total_rewards = 0

        while not(done or truncated):
            state = np.array(self.state)
            self.states_mem.append(state)
            action = self.action(np.array(self.states_mem, dtype=np.float32))

            
            
            next_state, reward, done, truncated, _ = self.env.step(action)
            total_rewards += reward
        
            self.state = next_state
            

            action = np.array(action)
            
            
            reward = np.array(reward)
            done = np.array(done, dtype=np.bool8)
            self.state = next_state
            show_img(state, t)

            t += 1 
        return t 

            
                


    def action(self, state, exploit = False):
        
        obs = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.network(obs.reshape(1, *obs.shape))
        action = np.argmax(q_values.detach().cpu().numpy())

        return action


import os
from PIL import Image
if __name__ == "__main__":
    env_name = "ALE/Breakout-v5"
    agent = Agent(env_name)
    duration = agent.play() 
    image_folder = 'img'
    gif_name = "result.gif"


    # Collect all image files from the folder
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(image_folder, file_name)
            images.append(Image.open(image_path))



    # Save the images as a GIF
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=399, loop=0)
    print(f"GIF saved as {gif_name}")


