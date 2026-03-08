# Unitree G1 Humanoid - Place Apple in Bowl (PPO)

Reinforcement learning project training a Unitree G1 humanoid robot to place an apple into a bowl using Proximal Policy Optimization (PPO). Built on the ManiSkill3 framework.

## Prerequisites

- Ubuntu (required for ManiSkill3)
- NVIDIA GPU with CUDA support (RTX 6000 Ada recommended)
- Python 3.10+

## Installation

### 1. System Dependencies

```bash
# Update package lists
sudo apt update

# Install ffmpeg (required for video rendering)
sudo apt install -y ffmpeg

# Optional: Make 'python' command available
sudo apt install -y python-is-python3

apt install python3-pip
```

### 2. Python Dependencies

```bash
# Install all Python packages (includes setuptools<70 for sapien compatibility)
pip install -r requirements.txt

pip install --upgrade mani_skill torch
```

### 3. Download Environment Asset

```bash
# Download the Unitree G1 robot and environment assets
python -m mani_skill.utils.download_asset UnitreeG1PlaceAppleInBowl-v1
```

## Usage

### Quick Start

```bash
# Launch Jupyter
jupyter notebook rl_humanoid_apple.ipynb
```

Then in the notebook:
1. ✅ **Run** Part 1: Setup and Imports
2. ❌ **Skip** Part 2: Environment Exploration (creates GPU PhysX conflict)
3. ✅ **Run** Part 3 onwards: Training

> ⚠️ **Important:** Do not run Part 2 before training. The exploration cells initialize GPU PhysX which can only be initialized once per Python session.

### Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named 'pkg_resources'` | Already fixed in requirements.txt (`setuptools<70`) |
| `ffmpeg not found` | Run `sudo apt install ffmpeg` |
| `GPU PhysX can only be enabled once` | Skip Part 2 in notebook, restart kernel |
| `mat1 and mat2 must have the same dtype` | Fixed - use updated notebook |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Total Timesteps | 5,000,000 |
| Parallel Envs (GPU) | 64 |
| Rollout Steps | 256 |
| Learning Rate | 3e-4 (linear decay) |
| Batch Size | 512 |
| Epochs per Update | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.005 |
| Value Function Coefficient | 0.5 |
| Max Grad Norm | 0.5 |
| Network | obs -> 512 -> 256 -> 256 (shared) |

## Expected Results

- Training time: 1-3 hours on RTX 6000 Ada
- Random baseline reward: approximately 0
- Trained model demonstrates clear task improvement and successful apple placement

## File Structure

```
├── README.md
├── requirements.txt
├── rl_humanoid_apple.ipynb       # Main training notebook
├── rl_humanoid_apple.py          # Training script export
```

## Credits

Built on [ManiSkill3](https://github.com/haosulab/ManiSkill), a unified benchmark for generalizable manipulation skills.
