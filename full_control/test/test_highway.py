import highway_env
import gymnasium as gym

env = gym.make("highway-v0")
obs, info = env.reset()
print("HighwayEnv installed successfully!")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)