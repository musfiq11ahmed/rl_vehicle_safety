import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import highway_env
from full_control.envs.highway_config_v2 import HIGHWAY_CONFIG

env = gym.make("highway-v0", config=HIGHWAY_CONFIG)
obs, info = env.reset()
print("Reset info:", info)

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("Step info:", info)
env.close()