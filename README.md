# Reentrant Network Environment with PPO

This repository contains the implementation of a Reentrant Network Environment using Proximal Policy Optimization (PPO) for job scheduling. The environment simulates a network of buffers and machines where packages are processed and moved between buffers.

## Files

- `low-traffic-5.py`: Implementation of the environment and PPO training loop for a low traffic scenario with 5 buffers and first buffer capacity of 250.
- `medium-traffic-5.py`: Implementation of the environment and PPO training loop for a medium traffic scenario with 5 buffers and first buffer capacity of 250.
- `high-traffic-5.py`: Implementation of the environment and PPO training loop for a high traffic scenario with 5 buffers and first buffer capacity of 250.
- `medium-traffic-5-infinite.py`: Implementation of the environment and PPO training loop for a medium traffic scenario with 5 buffers and infinite buffer capacity of 250.
- `medium-traffic-5-onebuffers.py`: Implementation of the environment and PPO training loop for a medium traffic scenario with 5 buffers where the first buffer capacity is 250 whereas all other buffers can hold one package each.
- `medium-traffic-10.py`: Implementation of the environment and PPO training loop for a medium traffic scenario with 10 buffers and first buffer capacity of 250.
- `medium-traffic-10-onebuffers.py`: Implementation of the environment and PPO training loop for a medium traffic scenario with 10 buffers and first buffer capacity of 250 and the sixth buffer a capacity of one.

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Gymnasium
- Matplotlib
