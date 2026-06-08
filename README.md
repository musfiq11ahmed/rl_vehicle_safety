# RL-Based Control Algorithm for Vehicular Safety

Reinforcement Learning-based control algorithm for autonomous vehicle safety, developed as part of research at NIRO Lab. Targets unstructured, high-density traffic environments representative of cities like Dhaka, Bangladesh.

---

## What this project does

Trains an RL agent to safely follow a lead vehicle in simulation using Time-to-Collision (TTC) as the primary safety metric. The agent learns to control acceleration and braking to avoid collisions while maintaining realistic driving speed.

---

## Current results (PPO baseline — 500k timesteps)

| Metric | Value |
|--------|-------|
| Collision rate | 0.0% |
| Episode completion rate | 100.0% |
| Mean TTC | 7.37s |
| TTC violation rate | 0.04% |
| Mean ego speed | 14.10 m/s |

---

## Project structure

```
rl_vehicle_safety/
├── envs/vehicle_env.py     # Custom Gymnasium MDP environment
├── agents/train_ppo.py     # PPO training script
├── evaluate.py             # Safety metrics evaluation
├── test_env.py             # Environment sanity check
└── test_install.py         # Library installation check
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install gymnasium stable-baselines3 matplotlib "numpy<2"
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python test_install.py
```

## Run

```bash
python test_env.py           # test environment
python agents/train_ppo.py   # train agent
python evaluate.py           # evaluate trained agent
```

---

## Roadmap

- [x] Phase 1 — Foundation and environment setup
- [x] Phase 2 — PPO baseline with safety-aware reward
- [ ] Phase 3 — HighwayEnv multi-agent simulation
- [ ] Phase 4 — Safe RL with formal safety constraints
- [ ] Phase 5 — Hardware validation


