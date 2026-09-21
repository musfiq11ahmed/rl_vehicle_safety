import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from full_control.envs.custom_env import VehicleSafetyEnv

# ── Callback: records reward after every episode ──────────────────────────────
class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self._current_rewards.append(reward)

        done = self.locals["dones"][0]
        if done:
            total = sum(self._current_rewards)
            self.episode_rewards.append(total)
            self.episode_lengths.append(len(self._current_rewards))
            self._current_rewards = []

            ep = len(self.episode_rewards)
            if ep % 50 == 0:
                recent = self.episode_rewards[-50:]
                print(f"Episode {ep:4d} | "
                      f"avg reward (last 50): {np.mean(recent):8.2f} | "
                      f"min: {np.min(recent):8.2f} | "
                      f"max: {np.max(recent):8.2f}")
        return True

# ── Environment setup ─────────────────────────────────────────────────────────
env = make_vec_env(VehicleSafetyEnv, n_envs=1)

# ── PPO Agent setup ───────────────────────────────────────────────────────────
model = PPO(
    policy         = "MlpPolicy",  # multi-layer perceptron policy network
    env            = env,
    learning_rate  = 3e-4,         # α — step size for gradient updates
    n_steps        = 2048,         # steps collected before each policy update
    batch_size     = 64,           # minibatch size for gradient computation
    n_epochs       = 10,           # how many times to reuse each batch
    gamma          = 0.99,         # γ — discount factor for future rewards
    gae_lambda     = 0.95,         # λ — GAE smoothing factor
    clip_range     = 0.2,          # ε — PPO clipping parameter
    ent_coef       = 0.01,         # entropy bonus — encourages exploration
    verbose        = 0,
    device         = "cpu"        # use your CPU
)

# ── Training ──────────────────────────────────────────────────────────────────
print("Starting PPO training on VehicleSafetyEnv...")
print(f"Device: {model.device}")
print("-" * 60)

logger = TrainingLogger()

model.learn(
    total_timesteps = 500_000,
    callback        = logger
)

# ── Save trained model ────────────────────────────────────────────────────────
os.makedirs("experiments", exist_ok=True)
model.save("experiments/ppo_vehicle_safety_v3")
print("\nModel saved to experiments/ppo_vehicle_safety.zip")

# ── Plot training curve ───────────────────────────────────────────────────────
rewards = logger.episode_rewards
window  = 50
rolling = [np.mean(rewards[max(0, i-window):i+1])
           for i in range(len(rewards))]

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, color="#534AB7", label="Episode reward")
plt.plot(rolling, color="#534AB7", linewidth=2,
         label=f"Rolling avg (window={window})")
plt.axhline(y=0,    color="gray",  linestyle="--", linewidth=0.8)
plt.axhline(y=1400, color="green", linestyle="--", linewidth=0.8,
            label="Target (1400)")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("PPO training — VehicleSafetyEnv")
plt.legend()
plt.tight_layout()
plt.savefig("experiments/training_curve.png", dpi=150)
plt.show()
print("Training curve saved to experiments/training_curve.png")