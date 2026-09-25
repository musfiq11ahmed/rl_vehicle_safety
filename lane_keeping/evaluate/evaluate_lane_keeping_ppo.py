# ─────────────────────────────────────────────────────────────────────────────
# evaluate/evaluate_lane_keeping_ppo.py
#
# Evaluates a trained PPO agent on lane-keeping-v0, for either the sub-stage
# 1.1 baseline env (s1_1) or the 1.2 action-rate-penalty env (s1_2),
# selected from the command line.
#
# Three corrections over a naive evaluation, all of which materially change
# the numbers:
#
#   1. Phase split. The vehicle starts 4 m off-centre, so the first seconds
#      are a recovery transient, not lane keeping. Averaging it together
#      with the settled portion inflates every tracking metric. Metrics are
#      reported separately for t < 6 s (recovery) and t >= 6 s (steady state).
#
#   2. Lane-switch exclusion. LaneKeepingEnv swaps its reference lane
#      mid-episode (self.lane = self.lanes.pop(0)). Lateral offset is then
#      measured against a different lane, producing a step discontinuity of
#      several metres that is an artefact of the reference change, not
#      vehicle motion. The step where the lane object changes identity is
#      dropped from the lateral and heading statistics.
#
#   3. Base vs penalised return. The action-rate penalty is recomputed here
#      from the recorded actions for BOTH sub-stages, so a 1.1 and a 1.2
#      policy can be scored on the same objective. Base return is what
#      compares against the 1.1 baseline (~198.4).
#
# Steering metrics (mean |steering change|, sign changes per episode) are
# the smoothness measures the built-in reward cannot see.
#
# Usage:
#   python evaluate_lane_keeping_ppo.py
#   python evaluate_lane_keeping_ppo.py --substage s1_2 --lam 0.05
#   python evaluate_lane_keeping_ppo.py --substage s1_2 --model <path.zip>
#
# All paths are resolved from this file's own location, so the script runs
# correctly from any working directory.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import argparse

# ── Path setup ─────────────────────────────────────────────────────────────────
# BASE_DIR is lane_keeping/evaluate/ ; adding its parent (lane_keeping/) to
# sys.path makes "envs.*" importable, matching the training scripts.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from highway_env.utils import wrap_to_pi

from envs.lane_keeping_config_s1_1 import make_lane_keeping_env, HIGHWAY_CONFIG
from envs.lane_keeping_config_s1_2 import make_lane_keeping_env_s1_2, DEFAULT_LAMBDA

EXP_DIR = os.path.join(BASE_DIR, "..", "experiments")

# Seconds per agent decision, used to convert step indices to real time.
DT = 1.0 / HIGHWAY_CONFIG["policy_frequency"]

# Boundary between the start-up transient and settled lane keeping. The
# 1.1 traces settle at roughly 6 s.
PHASE_SPLIT_S = 6.0

DPI = 300


# ── Command-line arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate a PPO lane-keeping policy (1.1 baseline or 1.2 penalised)."
)
parser.add_argument("--substage", choices=["s1_1", "s1_2"], default="s1_1",
                    help="s1_1 = baseline env, s1_2 = action-rate-penalty env. "
                         "Must match the env the model was trained on, since s1_2 "
                         "adds a 'prev_action' observation key.")
parser.add_argument("--model", default=None,
                    help="Path to the .zip model. Defaults to the unseeded "
                         "experiments/ model for the chosen sub-stage.")
parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                    help="Action-rate penalty weight. Used to build the s1_2 env "
                         "and to score the penalised return for both sub-stages.")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--sample-episode", type=int, default=0,
                    help="Index of the episode plotted as a time trace.")
parser.add_argument("--reward-history", default=None,
                    help="Path to the .npy training reward history for the "
                         "zoomed curve. Defaults to <model stem>_episode_rewards.npy "
                         "in experiments/.")
parser.add_argument("--zoom-ylim", nargs=2, type=float, default=[190.0, 200.0],
                    metavar=("YMIN", "YMAX"),
                    help="y-axis limits for the zoomed training curve.")
args = parser.parse_args()

# ── Resolve sub-stage-dependent paths and factory ──────────────────────────────
# The default models are the original single-seed runs, which were trained
# without a seed and are therefore labelled "_unseeded". Seeded runs
# (_seed0, _seed1, ...) are selected with --model.
if args.substage == "s1_1":
    default_model = os.path.join(EXP_DIR, "ppo_s1_1_unseeded.zip")
    make_env = lambda: make_lane_keeping_env()
else:
    default_model = os.path.join(EXP_DIR, f"ppo_s1_2_lam{args.lam}_unseeded.zip")
    make_env = lambda: make_lane_keeping_env_s1_2(lam=args.lam)

MODEL_PATH = args.model if args.model else default_model

# Output filenames are tagged with the model stem so runs do not overwrite
# each other's plots. splitext is avoided because a lambda such as 0.05
# makes ".05" look like the extension.
TAG = os.path.basename(MODEL_PATH)
if TAG.endswith(".zip"):
    TAG = TAG[:-4]

# Training scripts save every artifact of a run under the model's stem, so
# the reward history is derived from TAG. This keeps a --model pointing at a
# seeded run from silently plotting another run's training curve.
REWARD_HISTORY = (args.reward_history if args.reward_history
                  else os.path.join(EXP_DIR, f"{TAG}_episode_rewards.npy"))


# ── Per-step measurement helper ────────────────────────────────────────────────
def measure(env):
    """Returns (lateral_offset, heading_error) for the current vehicle state.

    lateral_offset : signed distance from the lane centerline in metres.
                     local_coordinates() projects the vehicle position onto
                     the lane and returns (longitudinal, lateral); the second
                     value is the tracking error the base reward penalises.
    heading_error  : signed angle in radians between the vehicle heading and
                     the lane tangent at that longitudinal position, wrapped
                     to [-pi, pi].
    """
    vehicle = env.unwrapped.vehicle
    lane    = env.unwrapped.lane
    longitudinal, lateral = lane.local_coordinates(vehicle.position)
    heading_error = wrap_to_pi(vehicle.heading - lane.heading_at(longitudinal))
    return lateral, heading_error


# ── Load model ─────────────────────────────────────────────────────────────────
env = make_env()
print(f"Sub-stage: {args.substage}  |  lambda: {args.lam}")
print(f"Loading model from {MODEL_PATH} ...")
try:
    model = PPO.load(MODEL_PATH, env=env)
except ValueError as exc:
    # The usual cause is a sub-stage mismatch: an s1_2 model carries the
    # extra "prev_action" observation key that the s1_1 env does not provide.
    print(f"\nFailed to load model into the {args.substage} environment:\n  {exc}")
    print("Check that --substage matches the env the model was trained on.")
    sys.exit(1)
print("Model loaded!\n")

# ── Evaluation rollouts ────────────────────────────────────────────────────────
# Signals are kept per episode so that differences never span a reset and
# phase masks can be applied per episode.
PHASES = ("recovery", "steady")
PHASE_LABELS = {
    "recovery": f"RECOVERY PHASE  (t < {PHASE_SPLIT_S:.0f} s)",
    "steady":   f"STEADY STATE    (t >= {PHASE_SPLIT_S:.0f} s)",
}

lat_by_phase          = {p: [] for p in PHASES}
head_by_phase         = {p: [] for p in PHASES}
steer_rate_by_phase   = {p: [] for p in PHASES}
sign_changes_by_phase = {p: [] for p in PHASES}
phase_steps_by_phase  = {p: [] for p in PHASES}   # steps per episode in phase

base_returns      = []
penalised_returns = []
switch_times      = []

sample_traces = None

for episode in range(args.episodes):
    obs, info = env.reset()
    laterals, headings, steerings, switched, base_rewards = [], [], [], [], []

    while True:
        # deterministic=True uses the policy mean rather than sampling, so
        # measured steering reflects the learned policy, not exploration noise.
        action, _ = model.predict(obs, deterministic=True)

        # action[0] is the normalised steering command in [-1, 1], mapping
        # linearly onto the configured steering_range (+/- 0.5236 rad).
        steerings.append(float(action[0]))

        # The reference lane is swapped inside LaneKeepingEnv.step(), so the
        # lane object is captured either side of the call to detect it.
        lane_before = env.unwrapped.lane
        obs, reward, terminated, truncated, info = env.step(action)
        switched.append(env.unwrapped.lane is not lane_before)

        # s1_2 reports the unpenalised reward in info; s1_1 has no penalty, so
        # the raw reward is already the base reward.
        base_rewards.append(float(info.get("base_reward", reward)))

        # Measured after the step, so each sample is the state produced by
        # the action recorded alongside it.
        lateral, heading_error = measure(env)
        laterals.append(lateral)
        headings.append(heading_error)

        if terminated or truncated:
            break

    lat_arr    = np.array(laterals)
    head_arr   = np.array(headings)
    steer_arr  = np.array(steerings)
    switch_arr = np.array(switched, dtype=bool)
    n          = len(lat_arr)
    time       = np.arange(n) * DT

    # Action-rate sequence used for scoring. a_prev = 0 before the first
    # action, matching ActionRatePenaltyWrapper, so for an s1_2 model at the
    # same lambda this reproduces the env's own penalised return exactly.
    deltas    = np.diff(np.concatenate(([0.0], steer_arr)))
    base_ret  = float(np.sum(base_rewards))
    base_returns.append(base_ret)
    penalised_returns.append(base_ret - args.lam * float(np.sum(deltas ** 2)))

    if switch_arr.any():
        switch_times.append(float(time[switch_arr][0]))

    masks = {"recovery": time < PHASE_SPLIT_S,
             "steady":   time >= PHASE_SPLIT_S}

    for phase, mask in masks.items():
        # Tracking metrics drop the lane-switch step (artefact, see header).
        valid = mask & ~switch_arr
        lat_by_phase[phase].append(lat_arr[valid])
        head_by_phase[phase].append(head_arr[valid])

        # Steering metrics use the contiguous phase slice. The lane switch
        # does not corrupt the commanded action, so no exclusion is applied.
        # np.diff here (unlike the scoring deltas above) measures true
        # step-to-step change, without the artificial 0 start.
        segment = steer_arr[mask]
        if segment.size > 1:
            steer_rate_by_phase[phase].append(np.abs(np.diff(segment)))
            signs = np.sign(segment)
            signs = signs[signs != 0]   # exact zeros would count as two flips
            sign_changes_by_phase[phase].append(int(np.sum(np.diff(signs) != 0)))
            phase_steps_by_phase[phase].append(segment.size)

    if episode == args.sample_episode:
        sample_traces = (lat_arr, head_arr, steer_arr, switch_arr, time, base_ret)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1:3d}/{args.episodes} | "
              f"base return: {base_ret:7.2f} | "
              f"penalised: {penalised_returns[-1]:7.2f} | "
              f"steps: {n}")

env.close()

# ── Metrics ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"  EVALUATION — {TAG}  ({args.episodes} episodes, sub-stage {args.substage})")
print("=" * 64)
print(f"  Mean base return        : {np.mean(base_returns):.2f} "
      f"+/- {np.std(base_returns):.2f}")
print(f"  Mean penalised return   : {np.mean(penalised_returns):.2f} "
      f"+/- {np.std(penalised_returns):.2f}   (lambda={args.lam})")
if switch_times:
    print(f"  Lane switch at          : {np.mean(switch_times):.2f} s "
          f"(1 step excluded per episode)")

for phase in PHASES:
    # Concatenate across episodes only at reporting time.
    lat  = np.concatenate(lat_by_phase[phase])
    head = np.concatenate(head_by_phase[phase])
    rate = np.concatenate(steer_rate_by_phase[phase])
    flips = np.array(sign_changes_by_phase[phase])

    print("\n" + "-" * 64)
    print(f"  {PHASE_LABELS[phase]}")
    print("-" * 64)
    # mean |lateral offset| : average distance held from the centerline.
    print(f"  Mean |lateral offset|   : {np.mean(np.abs(lat)):.4f} m")
    # RMS lateral offset : penalises large excursions more than the mean,
    # and is the quantity the base reward is a function of.
    print(f"  RMS lateral offset      : {np.sqrt(np.mean(lat ** 2)):.4f} m")
    # max |lateral offset| : worst single-step excursion in this phase.
    print(f"  Max |lateral offset|    : {np.max(np.abs(lat)):.4f} m")
    # mean |heading error| : angular misalignment with the lane tangent; a
    # policy can sit on the centerline while crabbing across it.
    print(f"  Mean |heading error|    : {np.mean(np.abs(head)):.4f} rad "
          f"({np.degrees(np.mean(np.abs(head))):.2f} deg)")
    # mean |steering change| : control smoothness. On a [-1, 1] action range,
    # values near 1 mean the command swings rail-to-rail every step.
    print(f"  Mean |steering change|  : {np.mean(rate):.4f} /step")
    # sign changes : chatter count. High values indicate oscillation rather
    # than steady control.
    print(f"  Steering sign changes   : {np.mean(flips):.1f} per episode "
          f"(over {np.mean(phase_steps_by_phase[phase]):.0f} steps)")

print("=" * 64)

# ── Plot: sample episode time traces ───────────────────────────────────────────
# Three stacked subplots sharing a time axis, so a lateral excursion can be
# read against the heading error and steering command that produced it.
lateral_trace, heading_trace, steering_trace, switch_trace, time_axis, sample_ret = sample_traces

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time_axis, lateral_trace, color="#534AB7", linewidth=1.5)
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Lateral offset [m]")
axes[0].set_title(f"Lane keeping — {TAG}, episode {args.sample_episode} "
                  f"(base return: {sample_ret:.2f})")

axes[1].plot(time_axis, np.degrees(heading_trace), color="#2E8B57", linewidth=1.5)
axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("Heading error [deg]")

axes[2].plot(time_axis, steering_trace, color="#C1440E", linewidth=1.5)
axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
axes[2].set_ylabel("Steering action [-1, 1]")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylim(-1.05, 1.05)

# Mark the phase boundary and the reference-lane switch, so the excluded
# sample and the transient/steady split are visible in the trace.
for ax in axes:
    ax.axvline(x=PHASE_SPLIT_S, color="black", linestyle=":", linewidth=1.2,
               label=f"phase split ({PHASE_SPLIT_S:.0f}s)")
    if switch_trace.any():
        ax.axvline(x=time_axis[switch_trace][0], color="darkorange",
                   linestyle=":", linewidth=1.2, label="lane switch")
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper right", fontsize=8)

plt.tight_layout()
sample_path = os.path.join(EXP_DIR, f"{TAG}_sample_episode.png")
plt.savefig(sample_path, dpi=DPI)
plt.show()
print(f"\nSample episode plot saved to {os.path.basename(sample_path)}")

# ── Plot: training curve, zoomed ───────────────────────────────────────────────
# Re-plots the saved training reward history with a clipped y-axis, where
# the converged portion of the curve sits (the base-reward ceiling is
# 200 = 200 steps x max reward 1.0). The full-range plot written during
# training compresses that band into a flat line.
if os.path.exists(REWARD_HISTORY):
    rewards = np.load(REWARD_HISTORY)
    window  = 50
    rolling = [np.mean(rewards[max(0, i - window):i + 1])
               for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color="#534AB7", label="Episode reward")
    plt.plot(rolling, color="#534AB7", linewidth=2,
             label=f"Rolling avg (window={window})")
    plt.ylim(args.zoom_ylim[0], args.zoom_ylim[1])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"PPO training — {TAG} "
              f"(zoomed: {args.zoom_ylim[0]:.0f}-{args.zoom_ylim[1]:.0f})")
    plt.legend()
    plt.tight_layout()
    zoom_path = os.path.join(EXP_DIR, f"{TAG}_training_curve_zoom.png")
    plt.savefig(zoom_path, dpi=DPI)
    plt.show()
    print(f"Zoomed training curve saved to {os.path.basename(zoom_path)}")
else:
    print(f"\nSkipped zoomed training curve: {os.path.basename(REWARD_HISTORY)} not found.")
    print("Re-run the matching training script to record the history.")
