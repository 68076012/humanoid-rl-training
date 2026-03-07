# Unitree G1 Humanoid - Place Apple in Bowl (PPO)

Reinforcement learning project training a Unitree G1 humanoid robot to place an apple into a bowl using Proximal Policy Optimization (PPO). Built on the ManiSkill3 framework.

## Prerequisites

- Ubuntu (required for ManiSkill3)
- NVIDIA GPU with CUDA support (RTX 6000 Ada recommended)
- Python 3.10+

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download the environment asset
python -m mani_skill.utils.download_asset UnitreeG1PlaceAppleInBowl-v1
```

## Usage

Launch the training notebook:

```bash
jupyter notebook rl_humanoid_apple.ipynb
```

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
├── rl_pushcube_lab.ipynb         # PushCube task notebook
├── rl_pushcube_lab.py            # PushCube script export
├── Robotics Simulation.ipynb     # General simulation tutorial
└── Robotics Simulation.py        # Tutorial script export
```

## Credits

Built on [ManiSkill3](https://github.com/haosulab/ManiSkill), a unified benchmark for generalizable manipulation skills.
