import gymnasium as gym 
import numpy as np
import keyboard
import time


import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
# Set up the environment


# Create the Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode = 'human')
env.reset()

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
    
    # Introduce a small delay to make it playable
    time.sleep(0.01)

env.close()