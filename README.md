# Reentrant Network Environment with PPO

This repository contains the implementation of a Reentrant Network Environment using Proximal Policy Optimization (PPO) for job scheduling. The environment models a network of buffers and machines, where packages are processed and transferred by a Gantry Robot through a series of work centers arranged in sequence.

## Files

- `ppo_five.py`: Implementation of the environment and PPO training loop for five work centers.
- `ppo_ten.py`: Implementation of the environment and PPO training loop for ten work centers.

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Gymnasium
- Matplotlib
