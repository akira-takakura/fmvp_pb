# Force-Modulated Visual Policy for Robot-Assisted Dressing with Arm Motions

## Arm-tracking integration in this fork

See [visible right-arm recording and offline PDE/GICP comparison](examples/arm_tracking/README.md).
The NumPy-only recorder can run in the existing simulator environment; the
numerical tracker is maintained in the linked companion repository. This
addition does not change the upstream policy or simulation dynamics.

Official codebase for the CoRL 2025 paper:

**Force-Modulated Visual Policy for Robot-Assisted Dressing with Arm Motions**  
Alexis Yihong Hao, Yufei Wang, Navin Sriram Ravie, Bharath Hegde, David Held*, Zackory Erickson*

---

## Overview
This repository contains the implementation of a **force-modulated visual policy** for robot-assisted dressing.

The method combines:
- Visual observations (point clouds)
- Force feedback from the gripper  
to enable robust dressing across **diverse arm motions**.

---

## Installation

### 1. Create environment
```bash
conda env create -f environment.yml
conda activate dressing
```

### 2. Install Assistive Gym (base environment)
This project builds on a modified version of Assistive Gym:

👉 https://github.com/Zackory/assistive-gym-fem

Please follow their installation instructions before running this code.

---

## Setup

Before running experiments:
```bash
source prepare.sh
```

This script sets up necessary paths and environment variables.

---

## Running Experiments

To evaluate trained checkpoints:
```bash
python scripts/eval_ckpts.py
```

---

## Project Structure (optional but recommended)
```
.
├── scripts/          # evaluation / training scripts
├── envs/             # simulation environments
├── algo/             # learning algorithms
├── reward/           # reward functions
├── prepare.sh        # environment setup
└── environment.yml   # conda dependencies
```

---

## Notes
- Sim-to-sim only release
This codebase includes the sim-to-sim transfer pipeline in PyBullet only.
The large-scale pretraining was conducted in a proprietary simulator, which cannot be released publicly.
- Garment dynamics
The garment simulation parameters in PyBullet are not fully optimized.
Performance can likely be improved by further tuning these parameters, and users are encouraged to explore this.

---

## Citation
If you find this work useful, please cite:
```
@inproceedings{hao2025force,
  title={Force-Modulated Visual Policy for Robot-Assisted Dressing with Arm Motions},
  author={Hao, Alexis Yihong and Wang, Yufei and Ravie, Navin Sriram and Hegde, Bharath and Held, David and Erickson, Zackory},
  booktitle={CoRL},
  year={2025}
}
```

---

## Contact
For questions, feel free to reach out or open an issue.
