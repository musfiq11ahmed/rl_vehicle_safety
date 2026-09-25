# ─────────────────────────────────────────────────────────────────────────────
# agents/train_lane_keeping_ppo_s1_1.py
#
# Sub-stage 1.1: trains a PPO agent (Stable-Baselines3) on HighwayEnv's
# built-in "lane-keeping-v0" environment, using the continuous-action config
# defined in envs/lane_keeping_config_s1_1.py.
#
# The TrainingLogger callback records per-episode reward and length during
# training so a reward curve can be plotted afterward. Once training
# finishes, the model, reward history and training curve are saved to
# lane_keeping/experiments/ under the run name ppo_s1_1_seed{SEED}. Paths
# are resolved from this file's location, so any working directory works.
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

from envs.lane_keeping_config_s1_1 import make_lane_keeping_env

# ── Seed ───────────────────────────────────────────────────────────────────────
# Seeds and parameter variants stay inside a sub-stage; each seed writes its
# own artifacts. Passed to PPO, which seeds Python, NumPy, PyTorch, the action
# space and the env, so a run is reproducible on the same machine.
SEED = 0


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
    seed          = SEED,
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

# Every artifact of this run shares one stem, so the evaluator can locate
# the matching reward history from the model path alone.
RUN_NAME = f"ppo_s1_1_seed{SEED}"
model.save(os.path.join(EXP_DIR, f"{RUN_NAME}.zip"))
print(f"\nModel saved to experiments/{RUN_NAME}.zip")

# Persist the raw per-episode rewards so the curve can be re-plotted later
# (e.g. zoomed) without re-running training.
np.save(os.path.join(EXP_DIR, f"{RUN_NAME}_episode_rewards.npy"),
        np.array(logger.episode_rewards))

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
plt.title(f"PPO training — {RUN_NAME} (lane-keeping-v0)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, f"{RUN_NAME}_training_curve.png"), dpi=300)
plt.show()
print("Training curve saved.")
