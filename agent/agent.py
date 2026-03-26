import gymnasium as gym
import numpy as np

# Load the environment
env = gym.make("gym_snakegame/SnakeGame-v0", board_size=10, render_mode="human")
