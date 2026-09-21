import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="human")
obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Initial observation:\n", obs)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()