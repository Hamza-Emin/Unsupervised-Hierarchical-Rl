# DIAYN Implementation - Your Own Version

This is your implementation of **DIAYN (Diversity is All You Need)** for unsupervised skill discovery using Soft Actor-Critic (SAC).

## 📁 Project Structure

```
MyOwn/
├── my_neural_nets.py      # Neural network architectures (Policy, Discriminator, Critic)
├── replay_buffer.py       # Experience replay buffer
├── dıayn_sac.py          # DIAYN-SAC agent implementation
├── train.py              # Training script
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium torch numpy
pip install gymnasium[mujoco]  # For HalfCheetah, Ant, etc.
```

### 2. Run Training

```bash
cd MyOwn
python train.py
```

This will:
- Train for 1000 episodes on HalfCheetah-v5
- Discover 10 different skills
- Evaluate skills every 50 episodes
- Save the trained agent to `diayn_agent.pth`

## 🎮 Customize Training

Edit the parameters in `train.py` at the bottom:

```python
agent, rewards = train_diayn(
    env_name='HalfCheetah-v5',      # Change environment
    num_episodes=1000,               # Number of training episodes
    max_steps_per_episode=1000,      # Steps per episode
    num_skills=10,                   # Number of skills to discover
    batch_size=256,                  # Training batch size
    updates_per_step=1,              # Gradient updates per step
    start_training_steps=1000,       # Random exploration steps
    eval_interval=50,                # Evaluation frequency
    device='cpu',                    # 'cpu' or 'cuda'
    render_mode=None,                # None or 'human' for visualization
    seed=42                          # Random seed
)
```

## 🎯 Available Environments

You can try different environments:
- `'HalfCheetah-v5'` - Fast bipedal robot (default)
- `'Ant-v5'` - Quadrupedal robot
- `'Walker2d-v5'` - Walking robot
- `'Hopper-v5'` - Single-legged hopper
- `'Humanoid-v5'` - Humanoid robot (complex!)

## 📊 Understanding the Output

### During Training

```
Episode 100/1000
  Skill: 3                              # Which skill was used
  Steps: 1000                           # Episode length
  Pseudo Reward: 125.45                 # DIAYN intrinsic reward
  Avg Pseudo Reward (last 10): 118.23   # Moving average
  Environment Reward: 2341.56           # External reward (not used for training)
  Total Steps: 100000                   # Total environment steps
  Discriminator Loss: 0.3421            # How well discriminator predicts skills
  Discriminator Accuracy: 0.9234        # % correct predictions (>90% is good!)
  Critic1 Loss: 12.45                   # Q-network losses
  Critic2 Loss: 11.89
  Policy Loss: -24.56                   # Policy improvement
  Alpha: 0.1834                         # Entropy coefficient (auto-tuned)
```

### Skill Evaluation

Every 50 episodes, you'll see:

```
SKILL EVALUATION AT EPISODE 50
Evaluating 10 skills...
  Skill  0: Avg Reward:  1234.56, Avg Displacement:   -12.34  # Moves backward
  Skill  1: Avg Reward:  2345.67, Avg Displacement:    45.67  # Moves forward
  Skill  2: Avg Reward:   345.89, Avg Displacement:     0.23  # Stays in place
  ...
  Skill  9: Avg Reward:  1567.89, Avg Displacement:    23.45

  Diversity Metrics:
    Reward std: 567.89          # Higher = more diverse rewards
    Displacement std: 23.45     # Higher = more diverse behaviors
```

## 🔑 Key Metrics

### Discriminator Accuracy
- **< 70%**: Skills not well-separated yet, keep training
- **70-90%**: Skills are becoming distinct
- **> 90%**: Skills are well-learned and diverse! ✅

### Pseudo Reward
- Should vary significantly across different skills
- Measures how distinctive each skill's states are

### Displacement Standard Deviation
- Measures diversity of movement patterns
- Higher = skills do more different things

## 💾 Save and Load Models

### Saving (automatic at end of training)
```python
from train import save_agent
save_agent(agent, 'my_agent.pth')
```

### Loading
```python
from train import load_agent
from dıayn_sac import DIAYN_SAC

# Create agent with same parameters
agent = DIAYN_SAC(state_dim=17, action_dim=6, skill_dim=10)

# Load trained weights
load_agent(agent, 'my_agent.pth', device='cpu')
```

## 🎨 Visualize Skills

To see the skills in action, set `render_mode='human'`:

```python
agent, rewards = train_diayn(
    env_name='HalfCheetah-v5',
    render_mode='human',  # Enable visualization
    num_episodes=100,     # Fewer episodes to watch
    ...
)
```

**Warning**: Rendering significantly slows down training!

## 🧪 Test Individual Skills

After training, test a specific skill:

```python
import gymnasium as gym

# Create environment
env = gym.make('HalfCheetah-v5', render_mode='human')

# Run skill 3
state, _ = env.reset()
for _ in range(1000):
    action = agent.select_action(state, skill=3, deterministic=True)
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## 🐛 Troubleshooting

### Low Discriminator Accuracy
- Train longer (more episodes)
- Increase `batch_size`
- Check that pseudo-rewards are non-zero

### Skills Look Similar
- Increase `num_skills`
- Train longer
- Try different environment

### Training is Slow
- Set `render_mode=None`
- Reduce `updates_per_step`
- Use GPU (`device='cuda'`)

### Out of Memory
- Reduce `batch_size`
- Reduce replay buffer capacity in `dıayn_sac.py`

## 📚 What You Learned

By implementing this, you now understand:

1. **Policy Networks**: Stochastic policies with Gaussian distributions
2. **Discriminator Networks**: Classification for skill identification
3. **Critic Networks**: Q-value estimation with twin networks
4. **SAC Algorithm**: Entropy-regularized reinforcement learning
5. **DIAYN**: Unsupervised skill discovery through mutual information
6. **Experience Replay**: Off-policy learning from stored transitions
7. **Target Networks**: Stable training with soft updates

## 🎓 Next Steps

1. **Experiment** with different environments
2. **Visualize** learned skills
3. **Tune** hyperparameters for better diversity
4. **Extend** to hierarchical RL (use skills for downstream tasks)
5. **Analyze** what each skill does

## 📖 References

- DIAYN Paper: "Diversity is All You Need" (Eysenbach et al., 2018)
- SAC Paper: "Soft Actor-Critic" (Haarnoja et al., 2018)

---

**Good luck with your skill discovery! 🚀**

