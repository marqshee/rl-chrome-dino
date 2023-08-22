# Mss used for screen capture
from mss import mss

# Sending commands
# import pydirectinput
import pyautogui

# Opencv allows us to do frame processing
import cv2

# Transformational framework
import numpy as np

# Optical Character Recognition (OCR) for game over extraction
import pytesseract

# Visualize captured frames
import matplotlib.pyplot as plt

# Bring in time for pauses
import time

# Environment components
import gymnasium as gym
# from gym import Env
# from gym.spaces import Box, Discrete

# File path management
import os

# Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

# Check env
from stable_baselines3.common import env_checker

from train import TrainAndLoggingCallback

# Import DQN algorithm
from stable_baselines3 import DQN

class WebGame(gym.Env):
    # Setup the env actin and observation shapes
    def __init__(self):
        # Subclass model
        super().__init__()
        
        # Setup spaces
        # Multidimensional space
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (1,83,100),
            dtype = np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        
        # Define extraction parameters for the game
        self.cap = mss()
        self.game_location = {
            'top': 300,
            'left': 0,
            'width': 600,
            'height': 500
        }
        self.done_location = {
            'top': 375,
            'left': 430,
            'width': 660,
            'height': 70
        }
    
    # What is called to do something in the game
    def step(self, action):
        # Action key - 0 = Space(jump), 1 - Duck(down), 2 = No Action (no op)
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pyautogui.press(action_map[action])
        
        # Checking whether the game is done
        terminated, done_cap = self.get_done()
        truncated = False
        # Get the next observation
        next_observation = self.get_observation()
        # Reward - we get a point for every frame we're alive
        reward = 1 # Define your own reward
        # Info dictionary
        info = {} # Stable baselines framework expects
        
        # obs, reward, terminated, truncated, info
        return next_observation, reward, terminated, truncated, info
    
    # Visualize the game
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            self.close()
    
    # This closes down the observation
    def close(self):
        cv2.destroyAllWindows()
    
    # Restart the game
    def reset(self, seed = None):
        # super().reset(seed=None)
        info = {}
        time.sleep(1)
        pyautogui.click(x = 150, y = 150)
        pyautogui.press('space')
        return self.get_observation(), info
        
    # Get the part of the game that we want
    def get_observation(self):
        # Get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, (100, 83)) # width x height
        # Add channels first
        channel = np.reshape(resized, (1,83,100)) # channel x height x width
        return channel
        
    # Get the done text
    def get_done(self):
        # Get done screen text using OCR
        done_cap = np.array(self.cap.grab(self.done_location))[:,:, :3] # all height, all width, all three channels
        # Valid done text
        done_strings = ['GAME', 'GAHE']
        
        # Apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True
        
        return done, done_cap

# Testing
env = WebGame()
# obs = env.get_observation()
# print(plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB)))
# # plt.show()
# done, done_cap = env.get_done()
# print(done)
# print(plt.imshow(done_cap))
# plt.show()

# Testing Loop - Play 10 games
# env = WebGame()
# for episode in range(2):
#     obs, info = env.reset()
#     terminated = False
#     total_reward = 0
    
#     while not terminated:
#         obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#         total_reward += reward
    
#     print(f'Total rewards for episode {episode} is {total_reward}')

# Create Callback
# env = WebGame()

# Check that the env is okay
env_checker.check_env(env)

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs'

callback = TrainAndLoggingCallback(check_freq = 1000, save_path = CHECKPOINT_DIR) 

# Create DQN model
model = DQN('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 1, buffer_size = 1200000, learning_starts = 1000, optimize_memory_usage=True)

# Kick off training
# model.learn(total_timesteps = 88000, callback = callback)
# model.load(os.path.join('train', 'best_model_88000'))
model = DQN.load(os.path.join('train', 'best_model_88000'))

# Testing Loop - Play 10 games
# env = WebGame()
for episode in range(1):
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        time.sleep(0.01)
        total_reward += reward
    
    print(f'Total rewards for episode {episode} is {total_reward}')
    time.sleep(2)