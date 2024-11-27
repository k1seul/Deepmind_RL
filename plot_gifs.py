from Agent import Agent
import argparse 

game = 'ALE/Pong-v5' 
network_weight = "data/ALE/Pong-v5/DQN_doubleddqn_1_step_800000.pt"

import torch
import numpy as np
from collections import deque
import random
from Env import wrap_deepmind, make_atari
from save_tools import save_numpy_to_gif
from types import SimpleNamespace
from networks import DQN

DUALNET = False
EVAL_EPISODES = 40
MAX_FRAME = 10_000 
ENV_NAME = 'ALE/Pong-v5' 

class Agent:
    def __init__(self) -> None:
        # Learning setting
        # Env init

        # self.env = make_env(self.env_name, clip_reward=True)
        env = make_atari(ENV_NAME)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env = wrap_deepmind(
            env, clip_rewards=False, frame_stack=False
        )
        self.n_actions = self.env.action_space.n
        args = {'frame_stack' : 4, 'device' : self.device, 'noisy_net' : False}
        args = SimpleNamespace(**args)
        self.frame_stack = 4 

        self._init_network(args)

    def _load_weights(self, weight_dir):
        self.network.load_state_dict(torch.load(weight_dir))

    def _init_network(self, args):
        
        if DUALNET:
            self.network = DuelDQN(n_actions=self.n_actions, args=args)
        else:
            self.network = DQN(n_actions=self.n_actions, args=args)
        
        self._load_weights(network_weight)
                
    def reset(self, env=None):
        if env is None:
            state, _ = self.env.reset()
        else:
            state, _ = env.reset()
        self.states_mem = deque(maxlen=self.frame_stack)
        state = np.array(state, dtype=np.uint8)
        for _ in range(self.frame_stack):
            self.states_mem.append(state)
        return state

    @torch.no_grad()
    def evaluate(self):
        ## evaluate the agent with fixed parameter and zero epsilon for 10 episodes

        # sum_rewards = 0
        rewards = []
        # env = make_env(self.env_name, clip_reward=False)
        env = make_atari(ENV_NAME)
        rendering = []
        env = wrap_deepmind(
            env, episode_life=False, clip_rewards=False, frame_stack=False
        )
        for i in range(EVAL_EPISODES):
            done = False
            truncated = False
            self.reset(env=env)
            rendered_imgs = []
            total_rewards = 0
            frame = 0
            while not (done or truncated) and frame < MAX_FRAME:
                frame += 1
                action = self.action(
                    np.array(self.states_mem, dtype=np.float32), exploit=True
                )
                next_state, reward, done, truncated, _ = env.step(action)
                total_rewards += reward
                self.states_mem.append(next_state)
                rendered_imgs.append(next_state)
            # sum_rewards += total_rewards
            rewards.append(total_rewards)
            rendering.append(rendered_imgs)

        return max(rewards), rendering[rewards.index(max(rewards))]

    def action(self, state, exploit=False):

        if random.random() < 0.05:
            action = random.randint(0, self.n_actions - 1)
        else:
            obs = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.network(obs.reshape(1, *obs.shape))
            action = np.argmax(q_values.detach().cpu().numpy())

        return action
    

if __name__ == '__main__':
    agent = Agent()
    rewards, render = agent.evaluate()
    # import pdb; pdb.set_trace()
    save_numpy_to_gif(np.array(render))
    print(rewards)