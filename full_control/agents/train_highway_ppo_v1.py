import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from full_control.envs.highway_config_v1 import make_highway_env


# ── Training logger callback ───────────────────────────────────────────────────
class HighwayLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        self._current_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            total = sum(self._current_rewards)
            self.episode_rewards.append(total)
            self.episode_lengths.append(len(self._current_rewards))
            self._current_rewards = []
            ep = len(self.episode_rewards)
            if ep % 50 == 0:
                recent = self.episode_rewards[-50:]
                print(f"Episode {ep:4d} | "
                      f"avg reward (last 50): {np.mean(recent):6.3f} | "
                      f"min: {np.min(recent):6.3f} | "
                      f"max: {np.max(recent):6.3f}")
        return True


# ── Environment setup ──────────────────────────────────────────────────────────
env = make_vec_env(make_highway_env, n_envs=1)

# ── PPO agent ─────────────────────────────────────────────────────────────────
model = PPO(
    policy        = "MlpPolicy",
    env           = env,
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 0,
    device        = "cpu"
)

# ── Training ───────────────────────────────────────────────────────────────────
print("Starting PPO training on HighwayEnv (continuous actions)...")
print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print("-" * 60)

logger = HighwayLogger()
model.learn(total_timesteps=200_000, callback=logger)

# ── Save model ─────────────────────────────────────────────────────────────────
os.makedirs("experiments", exist_ok=True)
model.save("experiments/ppo_highway_v1")
print("\nModel saved to experiments/ppo_highway_v1.zip")

# ── Plot training curve ────────────────────────────────────────────────────────
rewards = logger.episode_rewards
window  = 50
rolling = [np.mean(rewards[max(0, i-window):i+1])
           for i in range(len(rewards))]

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, color="#534AB7", label="Episode reward")
plt.plot(rolling, color="#534AB7", linewidth=2,
         label=f"Rolling avg (window={window})")
plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("PPO training — HighwayEnv (continuous actions)")
plt.legend()
plt.tight_layout()
plt.savefig("experiments/highway_training_curve_v1.png", dpi=150)
plt.show()
print("Training curve saved.")