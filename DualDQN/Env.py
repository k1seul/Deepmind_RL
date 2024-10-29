import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt 
import random 
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from collections import deque, namedtuple
from pympler import asizeof



class CropObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    Crop image wrapper 
    """
    def __init__(self, env: gym.Env, shape = (20, 104, 0, 84)) -> None:
        """
        crops the image based on shape
        The cropped images will have shape of 
        heights             width
        [shape[0]:shape[1], shape[2]:shape[3]]
        """
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

    
        assert len(shape) == 4 and all(
            x >= 0 for x in shape 
            ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

        self.crop_shape = tuple(shape)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to crop 

        Returns:
            The cropped observations
        """
        observation = observation
        observation = observation.astype('uint8')
        return observation[self.crop_shape[0]:self.crop_shape[1], self.crop_shape[2]:self.crop_shape[3]]
    

class ClipReward(gym.RewardWrapper, gym.utils.RecordConstructorArgs):
    """Clip rewards to 
        1 if positive 
        -1 else
        Args:
            env: The environment to apply the wrapper 
    """

    def __init__(self, env: gym.Env):
        """ Clip rewards to 
        1 if positive 
        -1 else
        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.RewardWrapper.__init__(self, env)


    def reward(self, reward):
        """
        Clip rewards to 
        1 if positive 
        -1 else
        """
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0
        return reward


    
    
# class ExperienceMemoryBuffer:
#     def __init__(self, maxlen=100_000):
#         self.mem = deque(maxlen=maxlen)
#         self.frame_num = 0

#     def add_experience(self, phi_t, a_t, r_t, phi_t_1, done):
#         self.mem.append((phi_t, a_t, r_t, phi_t_1, done))
#         self.frame_num += 1 

#     def __len__(self):
#         return len(self.mem)
    
#     def sample(self, batch_size=32):
#         if len(self) < batch_size:
#             return 
#         return np.array(random.sample(self.mem, batch_size))



# ## using python list
# class ExperienceMemoryBuffer:
#     def __init__(self, maxlen= 100_000):
#         self.maxlen = maxlen
#         self.mem_num = 0
#         self.stack_num = 4
        
#         self.states = [None] * self.maxlen
#         self.actions = [None] * self.maxlen
#         self.rewards = [None] * self.maxlen 
#         self.dones = [None]  * self.maxlen 


#     def add_experience(self, state, action, reward, done):
#         index = self.mem_num % self.maxlen
#         self.states[index] = state
#         self.actions[index] = action 
#         self.rewards[index] = reward 
#         self.dones[index] = done

#         self.mem_num += 1 
        

#     def __len__(self):
#         length = self.mem_num if self.mem_num < self.maxlen else self.maxlen
#         return length
    
#     def sample(self, batch_size=32):
#         if len(self) < batch_size:
#             return 
#         random_indices = random.sample(range(len(self) - 1), batch_size)
#         states_batch = [self.select_stack_states(i) for i in random_indices]
#         actions_batch = [self.actions[i] for i in random_indices]
#         rewards_batch = [self.rewards[i] for i in random_indices]
#         next_states_batch = [self.select_stack_states(i+1) for i in random_indices]
#         dones_batch = [self.dones[i] for i in random_indices]
#         # import pdb; pdb.set_trace()

#         return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch 

#     def select_stack_states(self, index):
#         indices = list(range(index - self.stack_num+1, index+1))
#         indices = [max(i, 0) for i in indices]

#         return np.stack([self.states[i] for i in indices])
    

class ExperienceMemoryBuffer:
    ### Numpy implementation
    def __init__(self, maxlen = 100_000, state_shape = (84, 84)):
        self.capacity = maxlen
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool8)
        self.position = 0 
        self.size = 0
        self.stack_num = 4

    def add_experience(self, state, action, reward, done):
        index = self.position % self.capacity
        self.states[index] = state
        self.actions[index] = action 
        self.rewards[index] = reward 
        self.dones[index] = done 

        self.position += 1 
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size = 32):
        indices = np.random.choice(self.size -1, batch_size, replace=False)
        next_state_indicies = indices + 1
        states = np.stack([self.select_stacked_states(i) for i in indices])
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        next_states = np.stack([self.select_stacked_states(i+1) for i in indices])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
    
    def select_stacked_states(self, index): 
        # import pdb; pdb.set_trace()
        indices = list(range(index - self.stack_num + 1, index + 1))
        indices = [max(i, 0) for i in indices]

        return np.stack([self.states[i] for i in indices])

    
            


        

def show_img(img, num):
    plt.figure()
    plt.imshow(img)
    plt.savefig('img/dummpy' + f'_{num}')

def make_env(env_name = "ALE/Breakout-v5"):
    env = gym.make(env_name)
    env = GrayScaleObservation(env) 
    env = ResizeObservation(env, (110, 84))
    env = CropObservation(env) 
    env = ClipReward(env)
    
    return env

def check_mem(env: gym.Env, mem_size = 100_000):
    ### env float32 -> uint8: 2.6gb -> 0.7gb 
    ### changed action, reward to uint8 -> 0.7gb almost same
    ### numpy replay memory buffer implementation -> 0.65Gb
    state , _ = env.reset()
    mem = ExperienceMemoryBuffer(mem_size)
    for _ in range(mem_size):
        action = action = random.randint(0, env.action_space.n-1) 
        state = np.array(state) 
        next_state, reward, done, truncated, _ = env.step(action)
        action = np.array(action, dtype=np.uint8)
        reward = np.array(reward, dtype=np.uint8)
        done = np.array(done, dtype=np.bool8)
        # import pdb; pdb.set_trace()
        mem.add_experience(state, action, reward, done)
        state = next_state
        if done or truncated:
            state, _ = env.reset()

    
    total_size = asizeof.asizeof(mem)
    print(f'Total memory usage of the list (including contents): {total_size/(1024 ** 3)} GB')
    

        

if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (110, 84))
    env = CropObservation(env)
    env = ClipReward(env)
    
    state, _ = env.reset() 
    for j in range(10):
        img, *_ =env.step(1)
        show_img(img, j-20)

    print(state.shape)
    # for i in range(100):
    #     action = random.randrange(2,4)
    #     # print(action)
    #     img, reward, done, trun, _ = env.step(action)
    #     # print(reward)
    #     # print(done or trun)
    #     show_img(img, i)
    # print(state)
    check_mem(env)

    

    # print(type(env))

    