# RL-Based Control Algorithm for Vehicular Safety

**Status: Work in progress**

This repository is part of an ongoing, funded research project at the NSU Intelligent Robotics (NIRO) Lab, North South University. It explores Reinforcement Learning-based control algorithms for autonomous vehicle safety in complex, real-world-representative traffic conditions.

As this is active research, some implementation details, results, and planned experiments are intentionally kept out of this README.

---

## Project structure

The project is organized into two tracks.

```
rl_vehicle_safety/
├── full_control/     # Full driving control experiments (baseline track)
│   ├── agents/         # Training scripts
│   ├── envs/            # Environment definitions and configs
│   ├── evaluate/         # Evaluation scripts
│   ├── visuals/            # Scripts to visualize trained agents
│   └── test/               # Sanity checks
│
├── lane_keeping/      # Current focus track
│   ├── agents/
│   ├── envs/
│   ├── evaluate/
│   ├── visuals/
│   └── experiments/
│
├── test_install.py
└── README.md
```

Trained models, result plots, and internal project notes are excluded from version control.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install gymnasium stable-baselines3 highway-env matplotlib agilerl "numpy<2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python test_install.py
```

---

## Acknowledgment

This work is conducted as part of a funded research initiative at the NSU Intelligent Robotics (NIRO) Lab.

---
