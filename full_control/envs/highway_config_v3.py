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

    "lanes_count"           : 3,
    "vehicles_count"        : 15,
    "duration"              : 40,
    "initial_spacing"       : 2,

    # Stay close to V1 values — only small targeted changes
    "collision_reward"      : -1.0,   # keep same as V1 — doubling broke things
    "right_lane_reward"     : 0.1,    # keep same as V1
    "high_speed_reward"     : 0.4,    # keep same as V1
    "lane_change_reward"    : -0.05,  # small penalty only — was 0.0 in V1
    "reward_speed_range"    : [20, 30],
    "simulation_frequency"  : 15,
    "policy_frequency"      : 5,
    "normalize_reward"      : True,  
}


def make_highway_env():
    env = gym.make("highway-v0", config=HIGHWAY_CONFIG)
    return env