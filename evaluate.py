import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from envs.vehicle_env import VehicleSafetyEnv

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_EPISODES = 100
MODEL_PATH   = "experiments/ppo_vehicle_safety_v3"

# ── Load model and environment ────────────────────────────────────────────────
env = VehicleSafetyEnv()
print(f"Loading model from {MODEL_PATH}.zip ...")
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded!\n")
print("=" * 60)
print(f"Evaluating over {NUM_EPISODES} episodes...")
print("=" * 60)

# ── Metric storage ────────────────────────────────────────────────────────────
episode_rewards            = []
episode_lengths            = []
collision_flags            = []
mean_ttc_per_episode       = []
min_ttc_per_episode        = []
ttc_violation_rate_per_episode = []
mean_speed_per_episode     = []
sample_ttc_trajectory      = []
sample_speed_trajectory    = []
sample_reward_trajectory   = []

# ── Evaluation loop ───────────────────────────────────────────────────────────
for episode in range(NUM_EPISODES):
    obs, info      = env.reset()
    episode_reward = 0.0
    episode_ttc_list   = []
    episode_speed_list = []
    episode_reward_list= []
    collided       = False
    steps          = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps          += 1
        episode_ttc_list.append(info["ttc"])
        episode_speed_list.append(info["ego_speed"])
        episode_reward_list.append(reward)

        if episode == 0:
            sample_ttc_trajectory.append(info["ttc"])
            sample_speed_trajectory.append(info["ego_speed"])
            sample_reward_trajectory.append(reward)

        if terminated:
            collided = True
            break
        if truncated:
            break

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    collision_flags.append(collided)
    mean_ttc_per_episode.append(np.mean(episode_ttc_list))
    min_ttc_per_episode.append(np.min(episode_ttc_list))
    violations = sum(1 for t in episode_ttc_list if t < 1.5)
    ttc_violation_rate_per_episode.append(violations / steps)
    mean_speed_per_episode.append(np.mean(episode_speed_list))

    if (episode + 1) % 20 == 0:
        recent_collisions = sum(collision_flags[-20:])
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | "
              f"reward: {episode_reward:8.2f} | "
              f"collisions (last 20): {recent_collisions} | "
              f"avg TTC: {np.mean(episode_ttc_list):.2f}s")

# ── Aggregate metrics ─────────────────────────────────────────────────────────
total_collisions    = sum(collision_flags)
collision_rate      = total_collisions / NUM_EPISODES
completion_rate     = 1.0 - collision_rate
mean_reward         = np.mean(episode_rewards)
std_reward          = np.std(episode_rewards)
mean_ttc            = np.mean(mean_ttc_per_episode)
mean_min_ttc        = np.mean(min_ttc_per_episode)
mean_violation_rate = np.mean(ttc_violation_rate_per_episode)
mean_ep_length      = np.mean(episode_lengths)
mean_speed          = np.mean(mean_speed_per_episode)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("       EVALUATION RESULTS — PPO VehicleSafetyEnv")
print("=" * 60)
print(f"  Episodes evaluated       : {NUM_EPISODES}")
print(f"  Total collisions         : {total_collisions}")
print("-" * 60)
print(f"  Collision rate           : {collision_rate*100:.1f}%")
print(f"  Episode completion rate  : {completion_rate*100:.1f}%")
print("-" * 60)
print(f"  Mean reward              : {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean avg TTC             : {mean_ttc:.3f} s")
print(f"  Mean min TTC (worst case): {mean_min_ttc:.3f} s")
print(f"  TTC violation rate       : {mean_violation_rate*100:.2f}%")
print("-" * 60)
print(f"  Mean episode length      : {mean_ep_length:.1f} / 200 steps")
print(f"  Mean ego speed           : {mean_speed:.2f} m/s")
print("=" * 60)

print("\nSafety assessment:")
if collision_rate == 0.0:
    print("  ✅ ZERO collisions — agent is fully safe")
elif collision_rate < 0.05:
    print(f"  ⚠️  Low collision rate ({collision_rate*100:.1f}%) — mostly safe")
elif collision_rate < 0.20:
    print(f"  ❌ Moderate collision rate ({collision_rate*100:.1f}%) — needs improvement")
else:
    print(f"  ❌ High collision rate ({collision_rate*100:.1f}%) — not converged")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("PPO Agent Evaluation — VehicleSafetyEnv", fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(episode_rewards, bins=30, color="#534AB7", alpha=0.8, edgecolor='white')
ax1.axvline(x=mean_reward, color='red',   linestyle='--', linewidth=1.5, label=f'Mean: {mean_reward:.0f}')
ax1.axvline(x=0,           color='gray',  linestyle='--', linewidth=0.8)
ax1.axvline(x=1400,        color='green', linestyle='--', linewidth=0.8, label='Target: 1400')
ax1.set_xlabel("Episode reward")
ax1.set_ylabel("Count")
ax1.set_title("Reward distribution")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(mean_ttc_per_episode, bins=30, color="#1D9E75", alpha=0.8, edgecolor='white')
ax2.axvline(x=1.5,     color='red',   linestyle='--', linewidth=1.5, label='Danger (1.5s)')
ax2.axvline(x=4.0,     color='green', linestyle='--', linewidth=1.5, label='Safe (4.0s)')
ax2.axvline(x=mean_ttc, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {mean_ttc:.2f}s')
ax2.set_xlabel("Mean TTC per episode (s)")
ax2.set_ylabel("Count")
ax2.set_title("TTC distribution across episodes")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 0])
timesteps = range(len(sample_ttc_trajectory))
ax3.plot(timesteps, sample_ttc_trajectory, color="#534AB7", linewidth=1.5, label="TTC (s)")
ax3.axhline(y=1.5, color='red',   linestyle='--', linewidth=1, label='Danger (1.5s)')
ax3.axhline(y=4.0, color='green', linestyle='--', linewidth=1, label='Safe (4.0s)')
ax3.fill_between(timesteps, 0, 1.5, alpha=0.1, color='red')
ax3.fill_between(timesteps, 4.0, 10, alpha=0.1, color='green')
ax3.set_xlabel("Timestep")
ax3.set_ylabel("TTC (seconds)")
ax3.set_title("Sample episode — TTC over time")
ax3.legend(fontsize=7)
ax3.set_ylim(0, 10.5)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(range(NUM_EPISODES), min_ttc_per_episode, color="#BA7517", alpha=0.6, linewidth=1, label="Min TTC per episode")
ax4.axhline(y=1.5,        color='red',  linestyle='--', linewidth=1.5, label='Danger (1.5s)')
ax4.axhline(y=mean_min_ttc, color='blue', linestyle='--', linewidth=1.5, label=f'Mean min: {mean_min_ttc:.2f}s')
collision_episodes = [i for i, c in enumerate(collision_flags) if c]
if collision_episodes:
    ax4.scatter(collision_episodes,
                [min_ttc_per_episode[i] for i in collision_episodes],
                color='red', s=30, zorder=5, label=f'Collisions ({len(collision_episodes)})')
ax4.set_xlabel("Episode")
ax4.set_ylabel("Min TTC (s)")
ax4.set_title("Worst-case TTC per episode")
ax4.legend(fontsize=7)

os.makedirs("experiments", exist_ok=True)
plt.savefig("experiments/evaluation_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nEvaluation plots saved to experiments/evaluation_results.png")