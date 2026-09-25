# ─────────────────────────────────────────────────────────────────────────────
# envs/lane_keeping_config.py
#
# Configuration and factory function for HighwayEnv's built-in
# "lane-keeping-v0" environment.
#
# lane-keeping-v0 is a low-level lateral control task: a single vehicle
# follows a sinusoidal reference lane using a bicycle dynamics model, and
# the only available action is steering (throttle/braking is disabled).
# This is a different control problem from highway-v0 — there is no
# surrounding traffic, no lane changing, and no discrete driving decisions,
# so the config keys below differ from full_control/envs/highway_config_v1.py.
#
# As with the full_control configs, the environment itself is implemented
# inside the installed highway_env library. This file only configures and
# instantiates it.
# ─────────────────────────────────────────────────────────────────────────────

import gymnasium as gym
import highway_env

# ── HighwayEnv configuration dictionary ────────────────────────────────────────
# Every key here configures part of the MDP that highway_env defines internally
# for LaneKeepingEnv (highway_env/envs/lane_keeping_env.py).
HIGHWAY_CONFIG = {
    # Action space: continuous steering only.
    # "longitudinal": False disables acceleration/braking control — speed is
    # fixed by the vehicle dynamics model, so the agent only steers.
    "action": {
        "type": "ContinuousAction",
        "steering_range": [-0.5236, 0.5236],   # +/- pi/6 rad, max steering angle
        "longitudinal": False,
        "lateral": True,
        "dynamical": True                       # use full bicycle dynamics model
    },

    # Observation space: vehicle state, its derivative, and the reference
    # state to track (lateral position and heading relative to the lane).
    "observation": {
        "type": "AttributesObservation",
        "attributes": ["state", "derivative", "reference_state"]
    },

    # Sensor noise added to the observed state and derivative, simulating
    # imperfect measurements.
    "state_noise": 0.05,
    "derivative_noise": 0.05,

    "simulation_frequency": 10,        # physics steps per second
    "policy_frequency": 10,            # agent decisions per second

    # Note: unlike highway-v0, this environment has no "duration" or
    # "collision_reward" keys — it never terminates or collides on its own
    # (there is no other traffic). Episode length is capped by the
    # max_episode_steps=200 set at gym registration for "lane-keeping-v0",
    # not by this config dict.

    # Rendering only, used by lane_keeping/visuals scripts.
    "screen_width": 600,
    "screen_height": 250,
    "scaling": 7,
    "centering_position": [0.4, 0.5],
}


# ── Environment factory function ────────────────────────────────────────────────
# make_vec_env() requires a callable that returns a fresh env instance
def make_lane_keeping_env():
    """Creates a configured lane-keeping-v0 instance for training/evaluation."""
    env = gym.make("lane-keeping-v0", config=HIGHWAY_CONFIG)
    return env
