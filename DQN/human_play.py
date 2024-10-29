import gym
import numpy as np
import cv2
from pynput import keyboard

# Set up the Atari environment
env = gym.make('AKE/Breakout-v5')
env.reset()

# Define actions
LEFT_ACTION = 4
RIGHT_ACTION = 5
NO_ACTION = 0

# Initialize variables
current_action = NO_ACTION

# Keyboard listener
def on_press(key):
    global current_action
    try:
        if key.char == 'a':  # Move left
            current_action = LEFT_ACTION
        elif key.char == 'd':  # Move right
            current_action = RIGHT_ACTION
    except AttributeError:
        if key == keyboard.Key.esc:  # Stop the game
            return False

def on_release(key):
    global current_action
    current_action = NO_ACTION

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Game loop
while True:
    # Get the next state and reward
    state, reward, done, *_ = env.step(current_action)
    
    # Render the environment
    env.render()

    # If the game is done, reset the environment
    if done:
        env.reset()
        print(f're game!')
    
    # Wait a short time to make the game playable
    cv2.waitKey(10)

# Clean up
env.close()
listener.stop()
