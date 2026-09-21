import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from full_control.envs.highway_config_v1 import HIGHWAY_CONFIG

MODEL_PATH = "experiments/ppo_highway_v1"

# Create environment WITH rendering enabled this time
env = gym.make("highway-v0", config=HIGHWAY_CONFIG, render_mode="human")

print(f"Loading model from {MODEL_PATH}.zip ...")
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded! Watch the window that opens...\n")

NUM_EPISODES_TO_WATCH = 5

for episode in range(NUM_EPISODES_TO_WATCH):
    obs, info = env.reset()
    episode_reward = 0
    steps = 0

    while True:
        # deterministic=True — agent always picks its best known action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        env.render()

        if terminated or truncated:
            crashed = info.get("crashed", False)
            status = "CRASHED" if crashed else "completed safely"
            print(f"Episode {episode+1}: {status} | "
                  f"steps: {steps} | reward: {episode_reward:.2f}")
            break

env.close()
print("\nDone watching trained agent.")