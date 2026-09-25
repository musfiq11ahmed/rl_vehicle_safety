# ─────────────────────────────────────────────────────────────────────────────
# envs/lane_keeping_config_s1_2.py
#
# Sub-stage 1.2 environment for the lane-keeping track: the 1.1 config is
# reused unchanged and an action-rate penalty is layered on top as a
# gymnasium wrapper.
#
# Motivation: the built-in reward of lane-keeping-v0 is
#     r_base = 1 - (lat / lane_width)^2
# which is a function of the vehicle state only — the action never enters
# it. Evaluation of 1.1 showed the consequence: the trained policy tracked
# the centerline well (0.38 m mean offset) while slamming the steering
# command rail-to-rail, flipping sign on ~70% of steps. The reward has no
# term that can see that, so nothing pushed the policy toward smooth
# control.
#
# This file adds that missing term:
#     r = r_base - LAMBDA * (a_t - a_prev)^2
# The squared action difference penalises large step-to-step changes in
# steering, which is the standard control-effort / actuator-smoothness
# regulariser. LAMBDA trades tracking accuracy against smoothness.
#
# The 1.1 module is imported, never modified, so baseline runs stay
# reproducible.
# ─────────────────────────────────────────────────────────────────────────────

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# HIGHWAY_CONFIG is imported as-is from 1.1 — the underlying MDP,
# observation type, action range and episode cap are all unchanged, so the
# only difference between 1.1 and 1.2 is the wrapper below.
from envs.lane_keeping_config_s1_1 import HIGHWAY_CONFIG

# Default penalty weight. At the 1.1 chatter level (mean |da| ~ 1.05 on
# a [-1, 1] action range) this costs roughly 0.055 reward per step, i.e.
# ~11 over a 200-step episode against a base return of ~198 — large enough
# to matter to the optimiser, small enough that tracking stays dominant.
DEFAULT_LAMBDA = 0.05


class ActionRatePenaltyWrapper(gym.Wrapper):
    """Penalises step-to-step change in the steering command.

    Also exposes the previous action to the agent as part of the
    observation. This is required, not cosmetic: the penalty term depends
    on a_prev, so without it in the observation the reward is not a
    function of the observed state and the MDP becomes partially
    observable — the agent would be punished for a quantity it cannot see,
    and the value function could not fit the penalty at all.
    """

    def __init__(self, env, lam=DEFAULT_LAMBDA):
        super().__init__(env)
        self.lam = lam

        # Previous action, held as float32 to match the action space dtype.
        self._prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Observation space gains a "prev_action" entry alongside the
        # existing state / derivative / reference_state keys. Bounds are
        # taken from the action space, so the normalised steering command
        # in [-1, 1] is described exactly.
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "prev_action": spaces.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                shape=env.action_space.shape,
                dtype=np.float32,
            ),
        })

    def _augment(self, obs):
        """Adds the stored previous action to a base observation dict."""
        augmented = dict(obs)
        augmented["prev_action"] = self._prev_action.copy()
        return augmented

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # a_prev starts at 0 each episode, i.e. the vehicle is assumed to
        # begin with a centred steering wheel. The first action is
        # therefore penalised relative to 0, which discourages an
        # immediate jump to full lock at the start of an episode.
        self._prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self._augment(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.env.action_space.shape
        )

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Squared L2 norm of the action change. Squared rather than absolute
        # so the gradient grows with the size of the jump, penalising
        # rail-to-rail swings far more than small corrections.
        delta = action - self._prev_action
        penalty = self.lam * float(np.sum(delta ** 2))
        reward = float(base_reward) - penalty

        # Both components are reported separately so evaluation can compare
        # base return against the 1.1 baseline (~198.4) while still
        # tracking what the agent actually optimised.
        info["base_reward"] = float(base_reward)
        info["action_rate_penalty"] = penalty

        # a_prev advances only after the penalty is computed, so the next
        # step measures the change relative to the action just applied.
        self._prev_action = action.copy()

        return self._augment(obs), reward, terminated, truncated, info


# ── Environment factory function ────────────────────────────────────────────────
# make_vec_env() requires a callable returning a fresh env; lam is forwarded
# through env_kwargs so a training run can sweep the penalty weight.
def make_lane_keeping_env_s1_2(lam=DEFAULT_LAMBDA):
    """Creates a lane-keeping-v0 instance with the action-rate penalty applied."""
    env = gym.make("lane-keeping-v0", config=HIGHWAY_CONFIG)
    return ActionRatePenaltyWrapper(env, lam=lam)
