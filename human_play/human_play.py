import gymnasium as gym 
import numpy as np
import keyboard
import time


import gymnasium as gym





# Create the Breakout environment
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """Constructor for the observation wrapper."""
        super().__init__(env)
        self.epi = 0

    def observation(self, observation):
            
        observation[2] = -observation[2]
        print(observation)

        return observation


def make_reverse_env(env="CartPole-v1", reversed=True):

    env = gym.make(env, render_mode='human')

    if reversed:
        env = ObservationWrapper(env)
    return env

env = make_reverse_env(reversed=False)
env.reset(seed=0)

# Define actions based on key presses
action_mapping = {
    'a': 0,  # NOOP
    'd': 1,  # FIRE
    'left': 2,  # LEFT
    'right': 3,  # RIGHT
}

# Start the game loop
done = False
while not done:
    env.render()
    
    # Default action is NOOP
    action = action_mapping['a']

    # Check for key presses
    if keyboard.is_pressed('left'):
        action = action_mapping['left']
    elif keyboard.is_pressed('right'):
        action = action_mapping['right']
    elif keyboard.is_pressed('d'):
        action = action_mapping['d']

    print(action)

    # Step the environment
    state, reward, done, *info = env.step(action)
    env.render()
    
    print(state)
    
    # Introduce a small delay to make it playable
    time.sleep(1)

env.close()