# ─────────────────────────────────────────────────────────────────────────────
# agents/train_lane_keeping_ppo_s1_2.py
#
# Sub-stage 1.2 training: PPO on lane-keeping-v0 with the action-rate
# penalty from envs/lane_keeping_config_s1_2.py applied.
#
# Identical to agents/train_lane_keeping_ppo_s1_1.py except that the environment
# is wrapped so the reward becomes
#     r = r_base - LAMBDA * (a_t - a_prev)^2
# and the observation carries the previous action.
#
# The logger tracks the penalised return (what PPO optimises) and the base
# return (r_base only) separately. The base return is the number directly
# comparable with the 1.1 baseline of ~198.4 — the penalised return is
# always lower by construction and says nothing on its own about whether
# tracking degraded.
#
# All artifacts share the run name ppo_s1_2_lam{LAMBDA}_seed{SEED}, so runs
# at different penalty weights or seeds do not overwrite each other.
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

from envs.lane_keeping_config_s1_2 import make_lane_keeping_env_s1_2

# ── Penalty weight and seed ────────────────────────────────────────────────────
# Both are variants inside sub-stage 1.2, not new sub-stages. Change them to
# sweep the tracking/smoothness trade-off or add seeds; output filenames pick
# the values up automatically. SEED is passed to PPO, which seeds Python,
# NumPy, PyTorch, the action space and the env.
LAMBDA = 0.05
SEED   = 0


# ── Training logger callback ───────────────────────────────────────────────────
class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []        # penalised return (the PPO objective)
        self.episode_base_rewards = []   # base return, comparable to baseline
        self.episode_lengths = []
        self._current_rewards = []
        self._current_base = []

    def _on_step(self) -> bool:
        self._current_rewards.append(self.locals["rewards"][0])

        # base_reward is injected by ActionRatePenaltyWrapper on every step.
        self._current_base.append(
            self.locals["infos"][0].get("base_reward", 0.0)
        )

        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_base_rewards.append(sum(self._current_base))
            self.episode_lengths.append(len(self._current_rewards))
            self._current_rewards = []
            self._current_base = []
            ep = len(self.episode_rewards)
            if ep % 50 == 0:
                recent      = self.episode_rewards[-50:]
                recent_base = self.episode_base_rewards[-50:]
                print(f"Episode {ep:4d} | "
                      f"avg penalised (last 50): {np.mean(recent):7.3f} | "
                      f"avg base (last 50): {np.mean(recent_base):7.3f} | "
                      f"min: {np.min(recent):7.3f} | "
                      f"max: {np.max(recent):7.3f}")
        return True


# ── Environment setup ──────────────────────────────────────────────────────────
# lam is forwarded to the factory through env_kwargs.
env = make_vec_env(make_lane_keeping_env_s1_2, n_envs=1,
                   env_kwargs={"lam": LAMBDA})

# ── PPO agent ─────────────────────────────────────────────────────────────────
# The observation is still a Dict (now with the extra "prev_action" key), so
# MultiInputPolicy remains required.
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
print(f"Starting PPO training on lane-keeping-v0 + action-rate penalty "
      f"(lambda={LAMBDA}, seed={SEED})...")
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
RUN_NAME = f"ppo_s1_2_lam{LAMBDA}_seed{SEED}"

# The ".zip" is written explicitly: LAMBDA values such as 0.05 put a dot in
# the stem, which stops Stable-Baselines3 from appending the extension
# itself (it would read ".05" as the suffix).
MODEL_PATH = os.path.join(EXP_DIR, f"{RUN_NAME}.zip")
model.save(MODEL_PATH)
print(f"\nModel saved to {os.path.basename(MODEL_PATH)}")

# Persist both reward histories so curves can be re-plotted later without
# re-running training.
np.save(os.path.join(EXP_DIR, f"{RUN_NAME}_episode_rewards.npy"),
        np.array(logger.episode_rewards))
np.save(os.path.join(EXP_DIR, f"{RUN_NAME}_episode_base_rewards.npy"),
        np.array(logger.episode_base_rewards))

# ── Plot training curve ────────────────────────────────────────────────────────
# Both series are drawn: the penalised return that PPO maximised, and the
# base return that is comparable with the 1.1 curve.
rewards = logger.episode_rewards
base    = logger.episode_base_rewards
window  = 50
rolling      = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
rolling_base = [np.mean(base[max(0, i-window):i+1])    for i in range(len(base))]

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, color="#534AB7", label="Episode reward (penalised)")
plt.plot(rolling, color="#534AB7", linewidth=2,
         label=f"Rolling avg penalised (window={window})")
plt.plot(rolling_base, color="#2E8B57", linewidth=2, linestyle="--",
         label=f"Rolling avg base (window={window})")
plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title(f"PPO training — {RUN_NAME} (lane-keeping-v0 + action-rate penalty)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, f"{RUN_NAME}_training_curve.png"), dpi=300)
plt.show()
print("Training curve saved.")
