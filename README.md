# DPAIL
Code for DPAIL: Training Diffusion Policy for Adversarial Imitation Learning without Policy Optimization

## Setup Instructions
To use this code, you need the following versions of Python and PyTorch:
Python >= 3.10
PyTorch >= 2.0.1 (install from the official website)

The Required Library
```
mujoco-py==2.1.2.14
gym==0.23.0
einops
```
### How to Use
```
python train_imitation.py --buffer buffers/Ant-v3/size10000 --cuda --
```
