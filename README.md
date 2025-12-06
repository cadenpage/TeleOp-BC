# TeleOp-BC: Behavioral Cloning from Teleoperation

A complete imitation learning system for robotic manipulation using behavioral cloning. This repository demonstrates learning box-pushing control policies from human demonstrations via keyboard, hand gesture tracking, or trained neural network policies.

## Overview

**TeleOp-BC** combines:
- **Physics Simulation** (MuJoCo): 1D box-pushing environment with realistic dynamics
- **Teleoperation Modes**: Keyboard, MediaPipe hand tracking, or pre-trained policies
- **Behavioral Cloning**: Neural network learns to mimic expert behavior from demonstrations
- **Real-time Visualization**: OpenCV-based rendering

## Key Capabilities

• Learn robot control policies from human demonstrations  
• Multi-modal teleoperation (keyboard, hand gestures, learned policies)  
• Physics-based simulation with real-time visualization  
• Complete training pipeline: collect data → train → evaluate  

## Setup

### Prerequisites
- Python 3.12
- Conda (Anaconda or Miniconda)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cadenpage/TeleOp-BC
   cd TeleOp-BC
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -n bc -f environment.yaml
   conda activate bc
   ```

3. **Verify installation:**
   ```bash
   conda env list  # Should show 'bc' environment
   python -c "import mujoco; import torch; print('Setup complete')"
   ```

## Usage

### 1. Collect Demonstrations via Keyboard
Use arrow keys to control the box:
```bash
python teleop.py --control keyboard --episodes 20 --render-mode human
```
Output: `demos_obs.npy` and `demos_act.npy`

### 2. Collect Demonstrations via Hand Gestures
Use thumb-index pinch distance for continuous force control:
```bash
python teleop.py --control mediapipe --episodes 20 --render-mode human --show-hand-debug
```

### 3. Train Behavioral Cloning Policy
```bash
python train_bc.py
```
Outputs: `data/policy.pt` (trained model checkpoint)

### 4. Evaluate with Trained Policy
Run the learned policy autonomously:
```bash
python teleop.py --control policy --policy-path data/policy.pt --episodes 8 --render-mode human
```

## File Structure

```
├── boxpush.py           # MuJoCo environment definition
├── boxpush.xml          # Physics model configuration
├── teleop.py            # Multi-modal control & data collection
├── train_bc.py          # Behavioral cloning training script
├── evaluation.py        # Policy evaluation utilities
├── environment.yaml     # Conda environment specification
├── .gitignore           # Git ignore patterns
└── data/                # Data directory (git-ignored)
    ├── demos_obs.npy    # Collected observation demonstrations
    ├── demos_act.npy    # Collected action demonstrations
    ├── policy_obs.npy   # Policy evaluation observations
    ├── policy_act.npy   # Policy evaluation actions
    └── policy.pt        # Trained policy checkpoint
```

## Packages Used

| Package | Purpose |
|---------|---------|
| **MuJoCo** | Physics simulation engine |
| **Gymnasium** | RL environment interface (reset/step API) |
| **PyTorch** | Neural network framework for policy learning |
| **MediaPipe** | Real-time hand pose estimation |
| **Pygame** | Keyboard input handling |
| **OpenCV** | Real-time visualization |
| **NumPy** | Numerical operations |

## Architecture

### Environment (boxpush.py)
- **Observation**: [block_position, block_velocity, relative_distance_to_target]
- **Action**: Continuous force [-1.0, 1.0] on the block
- **Episode Length**: 200 steps × 0.05s = 10 seconds
- **Task**: Push box to randomly sampled target position

### Policy Network
```
Input (3) → Dense(64) + ReLU → Dense(64) + ReLU → Output(1) + Tanh
```
- Trained with MSE loss on demonstration data
- Normalized observations during training and inference
- 10% validation split for monitoring

### Teleoperation Modes
1. **Keyboard**: Arrow keys map to [-1.0, 0.0, +1.0] force
2. **MediaPipe**: Pinch distance normalized to continuous [-1.0, +1.0]
3. **Policy**: Learned network predicts action from observation

## Example Workflow

```bash
# 1. Collect 10 mujoco demonstrations
python teleop.py --control mediapipe --episodes 10 --show-hand-debug
# Data saved to: data/demos_obs.npy, data/demos_act.npy

# 2. Train policy on collected data
python train_bc.py
# Model saved to: data/policy.pt

# 3. Test trained policy
python teleop.py --control policy --policy-path data/policy.pt --episodes 4
# Evaluation data saved to: data/policy_obs.npy, data/policy_act.npy
```

## Troubleshooting

**OpenCV window not showing on macOS:**
- The system uses offline rendering with OpenCV display
- Ensure X11/display server is properly configured
- Force headless mode: `--render-mode none`

**MuJoCo license warning:**
- Free 30-day trial; no impact on functionality
- For unlimited use, obtain free academic license from MuJoCo

**Memory issues:**
- Reduce `--episodes` or batch size in `train_bc.py`
- Use `render_mode none` to skip visualization

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)
- [MediaPipe Hands](https://github.com/google/mediapipe)

---

**Final Project for 396P** - Imitation Learning & Robotic Control
