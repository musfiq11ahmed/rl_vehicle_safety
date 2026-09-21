# ─────────────────────────────────────────────────────────────────────────────
# agents/train_lane_keeping_ppo.py
#
# Trains a PPO agent (Stable-Baselines3) on HighwayEnv's built-in
# "lane-keeping-v0" environment, using the continuous-action config defined
# in envs/lane_keeping_config.py.
#
# The TrainingLogger callback records per-episode reward and length during
# training so a reward curve can be plotted afterward. Once training
# finishes, the model and the training curve are saved to
# lane_keeping/experiments/ (assumes the script is run with the working
# directory set to lane_keeping/, matching full_control/agents/train_highway_ppo_v1.py).
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from envs.lane_keeping_config import make_lane_keeping_env


# ── Training logger callback ───────────────────────────────────────────────────
class TrainingLogger(BaseCallback):
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
env = make_vec_env(make_lane_keeping_env, n_envs=1)

# ── PPO agent ─────────────────────────────────────────────────────────────────
# "AttributesObservation" returns a Dict observation (state/derivative/
# reference_state), so PPO needs MultiInputPolicy instead of MlpPolicy —
# MlpPolicy only accepts a flat Box observation space.
model = PPO(
    policy        = "MultiInputPolicy",
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
print("Starting PPO training on lane-keeping-v0 (continuous actions)...")
print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print("-" * 60)

logger = TrainingLogger()
model.learn(total_timesteps=500_000, callback=logger)

# ── Save model ─────────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
os.makedirs(EXP_DIR, exist_ok=True)
model.save(os.path.join(EXP_DIR, "ppo_lane_keeping"))
print("\nModel saved to experiments/ppo_lane_keeping.zip")

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
plt.title("PPO training — lane-keeping-v0 (continuous actions)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, "lane_keeping_training_curve.png"), dpi=300)
plt.show()
print("Training curve saved.")
