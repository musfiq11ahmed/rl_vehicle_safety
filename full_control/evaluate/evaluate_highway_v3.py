import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from full_control.envs.highway_config_v3 import make_highway_env

NUM_EPISODES = 100
MODEL_PATH   = "experiments/ppo_highway_v2"

env = make_highway_env()
print(f"Loading model from {MODEL_PATH}.zip ...")
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print("Model loaded!\n")
print("=" * 60)
print(f"Evaluating over {NUM_EPISODES} episodes...")
print("=" * 60)

# ── Metric storage ─────────────────────────────────────────────────────────
episode_rewards          = []
episode_lengths          = []
crash_flags              = []
mean_speed_per_episode   = []
mean_high_speed_reward   = []
right_lane_rate          = []
sample_speed_trajectory  = []
sample_reward_trajectory = []

# ── Evaluation loop ────────────────────────────────────────────────────────
for episode in range(NUM_EPISODES):
    obs, info      = env.reset()
    episode_reward = 0.0
    speeds         = []
    hs_rewards     = []
    right_lane     = []
    crashed        = False
    steps          = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps          += 1
        speeds.append(info["speed"])
        hs_rewards.append(float(info["rewards"]["high_speed_reward"]))
        right_lane.append(float(info["rewards"]["right_lane_reward"]))

        if episode == 0:
            sample_speed_trajectory.append(info["speed"])
            sample_reward_trajectory.append(reward)

        if terminated:
            crashed = info.get("crashed", False)
            break
        if truncated:
            break

    episode_rewards.append(episode_reward)
    episode_lengths.append(steps)
    crash_flags.append(crashed)
    mean_speed_per_episode.append(np.mean(speeds))
    mean_high_speed_reward.append(np.mean(hs_rewards))
    right_lane_rate.append(np.mean(right_lane))

    if (episode + 1) % 20 == 0:
        recent_crashes = sum(crash_flags[-20:])
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | "
              f"reward: {episode_reward:7.2f} | "
              f"crashes (last 20): {recent_crashes} | "
              f"avg speed: {np.mean(speeds):.2f} m/s")

# ── Aggregate metrics ──────────────────────────────────────────────────────
total_crashes   = sum(crash_flags)
crash_rate      = total_crashes / NUM_EPISODES
mean_reward     = np.mean(episode_rewards)
std_reward      = np.std(episode_rewards)
mean_speed      = np.mean(mean_speed_per_episode)
mean_hs_reward  = np.mean(mean_high_speed_reward)
mean_right_lane = np.mean(right_lane_rate)
mean_length     = np.mean(episode_lengths)

# ── Print results ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("     EVALUATION RESULTS — PPO HighwayEnv V2")
print("=" * 60)
print(f"  Episodes evaluated     : {NUM_EPISODES}")
print(f"  Total crashes          : {total_crashes}")
print(f"  Crash rate             : {crash_rate*100:.1f}%")
print(f"  Mean reward            : {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean speed             : {mean_speed:.2f} m/s")
print(f"  Mean high speed reward : {mean_hs_reward:.3f}")
print(f"  Right lane rate        : {mean_right_lane*100:.1f}%")
print(f"  Mean episode length    : {mean_length:.1f} steps")
print("=" * 60)

print("\nSafety assessment:")
if crash_rate == 0.0:
    print("  ✅ ZERO crashes — agent is fully safe")
elif crash_rate < 0.05:
    print(f"  ⚠️  Low crash rate ({crash_rate*100:.1f}%) — mostly safe")
elif crash_rate < 0.20:
    print(f"  ❌ Moderate crash rate ({crash_rate*100:.1f}%) — needs improvement")
else:
    print(f"  ❌ High crash rate ({crash_rate*100:.1f}%) — not converged")

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("PPO Agent Evaluation — HighwayEnv V3", 
             fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1 — Reward distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(episode_rewards, bins=30, color="#534AB7", 
         alpha=0.8, edgecolor='white')
ax1.axvline(x=mean_reward, color='red', linestyle='--', 
            linewidth=1.5, label=f'Mean: {mean_reward:.1f}')
ax1.axvline(x=162.72, color='gray', linestyle='--', 
            linewidth=1, label='V1 baseline: 162.72')
ax1.set_xlabel("Episode reward")
ax1.set_ylabel("Count")
ax1.set_title("Reward distribution")
ax1.legend(fontsize=8)

# Plot 2 — Speed distribution across episodes
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(mean_speed_per_episode, bins=30, color="#1D9E75", 
         alpha=0.8, edgecolor='white')
ax2.axvline(x=mean_speed, color='blue', linestyle='--', 
            linewidth=1.5, label=f'Mean: {mean_speed:.1f} m/s')
ax2.axvline(x=17.93, color='gray', linestyle='--', 
            linewidth=1, label='V1 baseline: 17.93 m/s')
ax2.axvline(x=20.0, color='green', linestyle='--', 
            linewidth=1, label='Target min: 20 m/s')
ax2.set_xlabel("Mean speed per episode (m/s)")
ax2.set_ylabel("Count")
ax2.set_title("Speed distribution across episodes")
ax2.legend(fontsize=8)

# Plot 3 — Sample episode: speed and reward over time
ax3 = fig.add_subplot(gs[1, 0])
timesteps = range(len(sample_speed_trajectory))
ax3_twin  = ax3.twinx()
ax3.plot(timesteps, sample_speed_trajectory, 
         color="#534AB7", linewidth=1.5, label="Speed (m/s)")
ax3_twin.plot(timesteps, sample_reward_trajectory, 
              color="#BA7517", linewidth=1, alpha=0.7, label="Reward")
ax3.axhline(y=20.0, color='green', linestyle='--', 
            linewidth=1, label='Target speed (20 m/s)')
ax3.set_xlabel("Timestep")
ax3.set_ylabel("Speed (m/s)", color="#534AB7")
ax3_twin.set_ylabel("Reward", color="#BA7517")
ax3.set_title("Sample episode — speed and reward over time")
ax3.legend(loc='lower left', fontsize=7)
ax3_twin.legend(loc='lower right', fontsize=7)

# Plot 4 — Per episode: crash markers and speed over episodes
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(range(NUM_EPISODES), mean_speed_per_episode, 
         color="#1D9E75", alpha=0.6, linewidth=1, label="Mean speed")
ax4.axhline(y=mean_speed, color='blue', linestyle='--', 
            linewidth=1.5, label=f'Avg: {mean_speed:.1f} m/s')
ax4.axhline(y=20.0, color='green', linestyle='--', 
            linewidth=1, label='Target: 20 m/s')
crash_episodes = [i for i, c in enumerate(crash_flags) if c]
if crash_episodes:
    ax4.scatter(crash_episodes,
                [mean_speed_per_episode[i] for i in crash_episodes],
                color='red', s=50, zorder=5, 
                label=f'Crashes ({len(crash_episodes)})')
ax4.set_xlabel("Episode")
ax4.set_ylabel("Mean speed (m/s)")
ax4.set_title("Speed per episode with crash markers")
ax4.legend(fontsize=7)

os.makedirs("experiments", exist_ok=True)
plt.savefig("experiments/highway_evaluation_v3.png", dpi=150, 
            bbox_inches='tight')
plt.show()
print("\nEvaluation plots saved to experiments/highway_evaluation_v3.png")