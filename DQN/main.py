import torch 
from torch.nn import MSELoss
import numpy as np 
from Env import make_env, ExperienceMemoryBuffer
from network import DQN
from collections import deque
import random
from tensorboardX import SummaryWriter
import gc



class Agent:
    def __init__(self, writer,env_name = "ALE/Breakout-v5", device='cpu') -> None:
        self.mem = ExperienceMemoryBuffer(maxlen=1_000_000)
        self.env = make_env(env_name=env_name)
        self.n_actions = self.env.action_space.n
        self.network = DQN(n_actions=self.n_actions, device=device)
        self.device = device
        self.epsilon = 1.0
        self.epsilon_final = 0.1
        self.epsilon_decay_rate = 0.9 / 1_000_000
        self.loss = MSELoss() 
        self.optim = torch.optim.RMSprop(self.network.parameters(), lr=0.00025, alpha=0.95)
        self.gamma = 0.99
        self.total_frames = 10_000_000
        self.writer = writer
        self.td_loss = 0
        self.avg_q_value = 0 
        self.update_freq = 4
        self.target_net = DQN(n_actions=self.n_actions, device=device)
        self.target_net_update_freq = 10_000 

    def reset(self):
        self.state, _ = self.env.reset()
        self.states_mem = deque(maxlen=4)
        for _ in range(4):
            self.states_mem.append(self.state)
        # import pdb; pdb.set_trace()
        return self.state 
    
    def add_to_buffer(self):
        ### add 100_000 frames to buffer
        frame = 0
        print("adding frames to buffer")
        while frame < 10_000:
            done = False 
            truncated = False 
            self.reset()

            while not(done or truncated):
                state = np.array(self.state)
                action = random.randint(0, self.n_actions-1)
                next_state, reward, done, truncated, _ = self.env.step(action)
                

                action = np.array(action)
                reward = np.array(reward)
                done = np.array(done, dtype=np.bool8)
                self.mem.add_experience(state, action, reward, done)
                self.state = next_state 

                frame += 1
                if frame % 100 == 0:
                    print(f"Added: {frame} frames")


    def play(self, writer):
        t = 0
        episode_num = 0 
        self.add_to_buffer()
        while t < self.total_frames:
            done = False
            truncated = False 
            self.reset()
            total_rewards = 0
            episode_length = 0 

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
                self.mem.add_experience(state, action, reward, done)
                self.state = next_state 

                
                episode_length += 1 
                
                ## update network
                ## The mem.sample() returns None if len(mem) < batch_size
                update_samples = self.mem.sample()
                ## update samples < batch_size
                if update_samples is None:
                    continue
                if t % self.target_net_update_freq == 0:
                    self.target_net.load_state_dict(self.network.state_dict())
                if t % self.update_freq == 0:
                    # import pdb; pdb.set_trace()
                    self.update(update_samples)
                t += 1 

            
                

            if episode_num % 10 == 0:

                writer.add_scalar('data/rewards', total_rewards, episode_num )
                writer.add_scalar('data/epsilons', self.epsilon, episode_num )
                writer.add_scalar('data/steps', round(t/10_000_000 * 100, 1), episode_num )
                writer.add_scalar('data/episode_length', episode_length, episode_num )
                writer.add_scalar('data/td_error', self.td_loss, episode_num )
                writer.add_scalar('data/performace', self.evaluate_agent(), episode_num)
                writer.add_scalar('data/avg_q', self.avg_q_value, episode_num)
            print(f"Episode Number: {episode_num}, rewards: {total_rewards}, epsilon: {agent.epsilon}")
            episode_num += 1 

    def evaluate_agent(self):
        ## evaluate the agent with fixed parameter and zero epsilon for 10 episodes
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
                state = np.array(self.state)
                self.states_mem.append(state)
                # print(np.sum(np.array(self.states_mem, dtype=np.float32)))
                action = self.action(np.array(self.states_mem, dtype=np.float32), exploit=True)
                # import pdb; pdb.set_trace() 
                # print(action)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_rewards += reward
                
                self.state = next_state
                

            sum_rewards += total_rewards
        return sum_rewards / 10 



    def action(self, state, exploit = False):
        if random.random() < self.epsilon and not exploit:
            action = random.randint(0, self.n_actions - 1 )
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_final)
        else:    
            obs = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.network(obs.reshape(1, *obs.shape))
            action = np.argmax(q_values.detach().cpu().numpy())

        if not exploit:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_final)
        # if exploit:
        #     print(q_values)
        return action
    
    def update(self, samples):
        # print(samples)
        # import pdb; pdb.set_trace()
        # states = samples[0]
        states = np.stack(samples[0]) 
        # import pdb; pdb.set_trace()
        actions = np.stack(samples[1])
        rewards = np.stack(samples[2])
        next_states = np.stack(samples[3])
        
        dones = np.stack(samples[4])
        

        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.int16)

        Q_next_states = self.target_net(next_states).detach()
        # Q_next_states = self.network(next_states).detach() 
        Q_max_next_states = torch.max(Q_next_states, axis=1)[0] 
        y_i = rewards + self.gamma * Q_max_next_states * (1 - dones)
        Q_current_states = self.network(states)
        Q_current_states = Q_current_states[torch.arange(Q_current_states.size(0)), actions]


        loss = self.loss(Q_current_states, y_i)
        # loss.clamp_(min=-1, max=1)
        # print(loss)
        # import pdb; pdb.set_trace()
        
        self.optim.zero_grad()
        loss.backward() 
        self.optim.step()
        self.avg_q_value = torch.mean(Q_current_states).detach().cpu().numpy()
        self.td_loss = float(loss.detach().cpu().numpy() / 32)


if __name__ == '__main__':
    env_names = [
                 "ALE/Breakout-v5",
                 "ALE/BeamRider-v5",
                 "ALE/Enduro-v5",
                 "ALE/Pong-v5",
                 "ALE/Qbert-v5",
                 "ALE/Seaquest-v5",
                 "ALE/SpaceInvaders-v5"
                 ]

    for env in env_names:
        gc.collect()
        writer = SummaryWriter() 
        agent = Agent(env_name=env,writer=writer, device='cuda') 
        agent.play(writer)
        torch.save(agent.network.state_dict(), f'data/{env[4:]}.pt')
        writer.close() 
        