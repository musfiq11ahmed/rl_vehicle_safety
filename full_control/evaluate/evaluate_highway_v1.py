import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from full_control.envs.highway_config_v1 import make_highway_env

NUM_EPISODES = 100
MODEL_PATH   = "experiments/ppo_highway_v1"

env = make_highway_env()
print(f"Loading model from {MODEL_PATH}.zip ...")
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded!\n")

episode_rewards  = []
episode_lengths  = []
crash_flags      = []
mean_speed_list  = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    episode_reward = 0.0
    speeds = []
    steps = 0
    crashed = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        speeds.append(info.get("speed", 0))

        if terminated:
            crashed = info.get("crashed", False)
            break
        if truncated:
            break

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    crash_flags.append(crashed)
    mean_speed_list.append(np.mean(speeds))

    if (episode + 1) % 20 == 0:
        recent_crashes = sum(crash_flags[-20:])
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | "
              f"reward: {episode_reward:7.2f} | "
              f"crashes (last 20): {recent_crashes} | "
              f"avg speed: {np.mean(speeds):.2f}")

total_crashes  = sum(crash_flags)
crash_rate     = total_crashes / NUM_EPISODES
mean_reward    = np.mean(episode_rewards)
std_reward     = np.std(episode_rewards)
mean_speed     = np.mean(mean_speed_list)
mean_length    = np.mean(episode_lengths)

print("\n" + "=" * 60)
print("       EVALUATION RESULTS — PPO HighwayEnv")
print("=" * 60)
print(f"  Episodes evaluated  : {NUM_EPISODES}")
print(f"  Total crashes       : {total_crashes}")
print(f"  Crash rate          : {crash_rate*100:.1f}%")
print(f"  Mean reward         : {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean speed          : {mean_speed:.2f} m/s")
print(f"  Mean episode length : {mean_length:.1f} steps")
print("=" * 60)

plt.figure(figsize=(8, 5))
plt.hist(episode_rewards, bins=30, color="#534AB7", alpha=0.8, edgecolor='white')
plt.axvline(x=mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.1f}')
plt.xlabel("Episode reward")
plt.ylabel("Count")
plt.title(f"HighwayEnv PPO — Reward distribution (Crash rate: {crash_rate*100:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig("experiments/highway_evaluation_v1.png", dpi=600)
plt.show()
print("\nEvaluation plot saved.")