# Unsupervised Hierarchical Reinforcement Learning

Implementation of **DIAYN (Diversity is All You Need)** for unsupervised skill discovery using Soft Actor-Critic (SAC).

## 🎯 Overview

This project implements unsupervised skill discovery where an agent learns diverse behaviors **without any external reward signal**. The agent discovers skills by maximizing mutual information between skills and states.

## 📁 Project Structure

```
Unsupervised-Hierarchical-Rl/
├── DIAYN_soft_actor_critic/
│   ├── my_neural_nets.py      # Neural networks (Policy, Discriminator, Critic)
│   ├── replay_buffer.py       # Experience replay buffer
│   ├── dıayn_sac.py          # DIAYN-SAC agent implementation
│   ├── train.py              # Training script
│   ├── inference.py          # Run & visualize trained skills
│   ├── record_video.py       # Record skill videos
│   └── README.md             # Detailed documentation
├── diayn_agent.pth           # Trained agent (Ant-v5, hidden_dim=256)
├── diayn_agent2.pth          # Trained agent (Ant-v5, hidden_dim=128)
├── diayn_agent3.pth          # Trained agent (HalfCheetah-v5, hidden_dim=128)
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install gymnasium torch numpy
pip install gymnasium[mujoco]
```

### Training

```bash
cd DIAYN_soft_actor_critic
python train.py
```

### Inference (Run trained skills)

```bash
# Interactive mode
python inference.py --checkpoint ../diayn_agent3.pth --env HalfCheetah-v5 --mode interactive

# Visualize all skills
python inference.py --checkpoint ../diayn_agent3.pth --env HalfCheetah-v5 --mode visualize

# Run single skill
python inference.py --checkpoint ../diayn_agent3.pth --env HalfCheetah-v5 --mode single --skill 5
```

### Record Videos

```bash
python record_video.py
```

## 🤖 Trained Agents

| Agent | Environment | Hidden Dim | Command |
|-------|-------------|------------|---------|
| `diayn_agent.pth` | Ant-v5 | 256 | `--env Ant-v5 --hidden-dim 256` |
| `diayn_agent2.pth` | Ant-v5 | 128 | `--env Ant-v5 --hidden-dim 128` |
| `diayn_agent3.pth` | HalfCheetah-v5 | 128 | `--env HalfCheetah-v5 --hidden-dim 128` |

## 🧠 Algorithm

DIAYN discovers diverse skills by maximizing mutual information:

```
I(S; Z) = H(Z) - H(Z|S)
```

### Neural Networks

- **Policy Network**: Takes state + skill (one-hot) → outputs action distribution
- **Discriminator**: Takes state → predicts which skill caused it (classifier)
- **Twin Critics**: Estimate Q-values for (state, action, skill) tuples

### Training Loop

1. Sample a random skill z
2. Policy takes actions conditioned on skill
3. Discriminator tries to identify skill from resulting states
4. Pseudo-reward = log q(z|s') - log p(z)
5. Update all networks using SAC

## 📊 Key Metrics

- **Discriminator Accuracy**: >90% means skills are well-separated
- **Displacement Std**: Higher = more diverse movement patterns
- **Pseudo Reward**: Measures how distinctive each skill's states are

## 📚 References

- [DIAYN: Diversity is All You Need (Eysenbach et al., 2018)](https://arxiv.org/abs/1802.06070)
- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)

## 👤 Author

Hamza Emin

