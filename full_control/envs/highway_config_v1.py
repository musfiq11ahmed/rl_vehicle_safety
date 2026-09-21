# ─────────────────────────────────────────────────────────────────────────────
# envs/highway_config.py
# Configuration and factory function for HighwayEnv
# Unlike vehicle_env.py, we do not define the MDP from scratch here —
# highway_env (the installed library) already implements the environment.
# This file only configures and instantiates it.
# ─────────────────────────────────────────────────────────────────────────────

import gymnasium as gym
import highway_env

# ── HighwayEnv configuration dictionary ────────────────────────────────────────
# Every key here configures part of the MDP that highway_env defines internally
HIGHWAY_CONFIG = {
    # Action space: continuous [steering, acceleration] — matches Phase 4 SAC plan
    "action": {
        "type": "ContinuousAction"
    },

    # Observation space: 5 vehicles × 5 features [presence, x, y, vx, vy]
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True
    },

    "lanes_count": 3,                  # 3-lane highway
    "vehicles_count": 15,              # 15 surrounding vehicles
    "duration": 40,                    # episode length in seconds
    "initial_spacing": 2,              # initial vehicle spacing

    # Built-in reward weights (highway_env computes reward internally)
    "collision_reward": -1.0,          # penalty for collision
    "right_lane_reward": 0.1,          # reward for keeping right lane
    "high_speed_reward": 0.4,          # reward for driving fast
    "lane_change_reward": 0.0,         # no reward/penalty for lane changes
    "reward_speed_range": [20, 30],    # target speed range (m/s)

    "simulation_frequency": 15,
    "policy_frequency": 5,
}


# ── Environment factory function ────────────────────────────────────────────────
# make_vec_env() requires a callable that returns a fresh env instance
def make_highway_env():
    """Creates a configured HighwayEnv instance for training/evaluation."""
    env = gym.make("highway-v0", config=HIGHWAY_CONFIG)
    return env