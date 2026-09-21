import gymnasium as gym
import highway_env

HIGHWAY_CONFIG = {
    "action": {
        "type": "ContinuousAction"
    },
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True
    },

    "lanes_count"     : 3,
    "vehicles_count"  : 15,
    "duration"        : 40,
    "initial_spacing" : 2,

    # ── Reward weights ──────────────────────────────────────────────────────
    # Increased collision penalty — was -1.0, now -2.0
    # Makes safety more non-negotiable relative to other rewards
    "collision_reward"      : -2.0,

    # Reduced right lane reward — was 0.1, now 0.05
    # Less pull toward rightmost lane reduces unnecessary lane changes
    "right_lane_reward"     : 0.05,

    # Increased speed reward — was 0.4, now 0.6
    # Stronger incentive to drive fast encourages overtaking
    "high_speed_reward"     : 0.6,

    # Lane change penalty — was 0.0, now -0.1
    # Discourages unnecessary lane changes
    "lane_change_reward"    : -0.1,

    "reward_speed_range"    : [20, 30],
    "simulation_frequency"  : 15,
    "policy_frequency"      : 5,
    "normalize_reward"      : True,
}


def make_highway_env():
    env = gym.make("highway-v0", config=HIGHWAY_CONFIG)
    return env