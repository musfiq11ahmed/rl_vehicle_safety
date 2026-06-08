import gymnasium as gym
import stable_baselines3
import torch
import numpy as np
import matplotlib.pyplot as plt

print("All libraries imported successfully!")
print("Python torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

env = gym.make("CartPole-v1")
obs, info = env.reset()
print("CartPole state shape:", obs.shape)
print("Setup complete — ready to build!")