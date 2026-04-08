# Reinforcement Learning & Control

This document curates state-of-the-art learning algorithms tailored for robotics, particularly focusing on sim-to-real transfer and high-dimensional control.

---

## [FlashSAC](https://holiday-robot.github.io/FlashSAC/)

**FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control**

A fast and stable off-policy RL algorithm that achieves the highest asymptotic performance in the shortest wall-clock time among existing methods for high-dimensional sim-to-real robotic control.

### Why FlashSAC?
While PPO has historically been the default for sim-to-real RL, it relies on discarding past transitions after every update because it is on-policy. This makes it inefficient for complex formulations like humanoids, dexterous manipulation, or vision-based control.

Off-policy methods are more sample-efficient but notoriously unstable and slow when scaled up due to compounding bootstrapping errors. FlashSAC solves this through three main pillars:

1. **Fast Training**: Trades frequent small updates for massive data throughput. It uses up to 1024 parallel environments, huge replay buffers (10M transitions), large model sizes (2.5M parameters), and highly optimized code (JIT compilation & mixed precision).
2. **Stable Training**: Uses architectural constraints like Inverted Residual Backbones and Pre-activation Batch Normalization. It also limits weight, feature, and gradient norms dynamically, preventing the bootstrapping errors from compounding.
3. **Broad Exploration**: Introduces a unified entropy target mathematically consistent across various embodiments without needing per-task tuning, accompanied by action noise repetition to ensure temporally correlated exploration.

**Result**: It slashes sim-to-real humanoid training from hours to mere minutes, outperforming existing algorithms across 60+ tasks in 10 different simulators.

### Resources
- [Project Website](https://holiday-robot.github.io/FlashSAC/)
- [Paper (arXiv)](https://arxiv.org/abs/2604.04539)
- [GitHub Repository](https://github.com/Holiday-Robot/FlashSAC)

---

## [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

**Stable Baselines3: Reliable Reinforcement Learning Implementations in PyTorch**

Stable Baselines3 (SB3) is a robust and highly trusted set of PyTorch implementations of reinforcement learning algorithms. Designed with a focus on code simplicity and comprehensive documentation, it serves as an excellent foundation for research and rapid prototyping.

### Supported Algorithms
SB3 focuses on standard and reliable RL algorithms:
- **A2C** (Advantage Actor Critic): A synchronous, deterministic variant of A3C.
- **DDPG** (Deep Deterministic Policy Gradient): An off-policy algorithm for continuous action spaces.
- **DQN** (Deep Q-Network): An off-policy algorithm for discrete action spaces, featuring extensions like Double DQN and Dueling DQN.
- **HER** (Hindsight Experience Replay): An algorithm that can be combined with off-policy methods effectively for goal-conditioned tasks.
- **PPO** (Proximal Policy Optimization): The versatile default on-policy algorithm with clipped surrogate objectives, balancing sample efficiency and simple tuning.
- **SAC** (Soft Actor-Critic): A state-of-the-art off-policy maximum entropy actor-critic algorithm.
- **TD3** (Twin Delayed DDPG): A more stable continuous-action off-policy algorithm addressing over-estimation bias in DDPG.

**Note on Integrations**: SB3 models plug easily into common simulation ecosystems, providing `gymnasium` support out-of-the-box. There is also **SB3 Contrib** for experimental algorithms (like Maskable PPO, QRDQN, Recurrent PPO) and **SBX** for fast JAX-based variants.

### Resources
- [GitHub Repository](https://github.com/DLR-RM/stable-baselines3)
- [Official Documentation](https://stable-baselines3.readthedocs.io/en/master/)

---

## [RLMujoco](https://github.com/Jitu0110/RLMujoco)

**Applied RL on MuJoCo via Gymnasium and Stable Baselines3**

RLMujoco is an applied project that implements and investigates Deep Reinforcement Learning training specifically on MuJoCo environments (such as *Humanoid-v4*, *Ant-v4*, and *HalfCheetah-v4*) using the Gymnasium interface and Stable Baselines3.

### Key Focus Areas
- **Standardized Implementations**: Shows how to cleanly implement SAC, PPO, A2C, and DQN on physical robot models.
- **Reward Manipulation**: Demonstrates how to edit sub-environment files to manipulate reward computations (e.g., heavily weighting the `forward_reward` or penalizing `ctrl_cost`) to generate specific behaviors like high-speed forward locomotion.
- **Model Tuning**: Serves as a great baseline template for exploring hyperparameters (learning rates, discount factors) and inspecting the results dynamically utilizing TensorBoard logs.

### Resources
- [GitHub Repository](https://github.com/Jitu0110/RLMujoco)
