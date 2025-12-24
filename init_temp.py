import gymnasium as gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from collections import deque
import copy


# ==================== Replay Buffer ====================
class ReplayBuffer:
    
    
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, skill):

        self.buffer.append((state, action, reward, next_state, done, skill))
    
    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(skills)
        )
    
    def __len__(self):
        return len(self.buffer)



class Policy(nn.Module):

    def __init__(self, input_dim, skill_dim=10, hidden_dim=256, action_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.skill_dim = skill_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Input layer: observation + skill embedding
        self.fc1 = nn.Linear(input_dim + skill_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for mean and std of actions
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observation, skill):
       
        x = torch.cat([observation, skill], dim=-1)
        
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
      
        action_mean = self.mean_layer(x)
        log_std = self.std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        action_std = log_std.exp()
        
        return action_mean, action_std
    
    def sample(self, observation, skill):
        """
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            mean: Mean of the distribution
        """
        mean, std = self.forward(observation, skill)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
     
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        mean = torch.tanh(mean)
        
        return action, log_prob, mean


class Discriminator(nn.Module):

    def __init__(self, input_dim=17, hidden_dim=256, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
       
        self.relu = nn.ReLU()
        
     
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observation):
        """        
        Returns:
            skill_probs: Probability distribution over skills [batch_size, output_dim]
        """

        x = self.relu(self.layer_1(observation))
        x = self.relu(self.layer_2(x))
        
      
        logits = self.output(x)
        
        
        return logits

class Critic(nn.Module):

    def __init__(self, input_dim, action_dim, skill_dim=10, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.hidden_dim = hidden_dim

        # Input layer: observation + action + skill
        self.fc1 = nn.Linear(input_dim + action_dim + skill_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: single Q-value
        self.output = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observation, action, skill_one_hot):
        """
        Forward pass through the critic network
        
        Args:
            observation: State observation tensor [batch_size, input_dim]
            action: Action tensor [batch_size, action_dim]
            skill_one_hot: One-hot skill vector tensor [batch_size, skill_dim]
        
        Returns:
            q_value: Q-value estimate [batch_size, 1]
        """
        # Concatenate observation, action, and skill
        x = torch.cat([observation, action, skill_one_hot], dim=-1)
        
        # Pass through hidden layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Output Q-value
        q_value = self.output(x)
        
        return q_value


# ==================== DIAYN SAC Agent ====================
class DIAYN_SAC:
    """DIAYN algorithm with SAC as the base RL algorithm"""
    
    def __init__(
        self,
        state_dim=17,
        action_dim=6,
        skill_dim=10,
        hidden_dim=256,
        lr_policy=3e-4,
        lr_critic=3e-4,
        lr_discriminator=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Initialize networks
        self.policy = Policy(state_dim, skill_dim, hidden_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, skill_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, skill_dim, hidden_dim).to(device)
        self.discriminator = Discriminator(state_dim, hidden_dim, skill_dim).to(device)
        
        # Target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Freeze target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)
        
        # Entropy coefficient (can be tuned automatically)
        self.alpha = alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_policy)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        
        # For uniform skill prior
        self.log_skill_prior = -np.log(skill_dim)
    
    def select_action(self, state, skill, deterministic=False):
       
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        skill_onehot = torch.zeros(1, self.skill_dim).to(self.device)
        skill_onehot[0, skill] = 1.0
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state, skill_onehot)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.policy.sample(state, skill_onehot)
        
        return action.cpu().numpy()[0]
    
    def compute_pseudo_reward(self, next_state, skill):
        """
        Compute DIAYN pseudo-reward: r(s,z) = log q(z|s') - log p(z)
        where p(z) is uniform, so log p(z) = -log(k)
        """
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.discriminator(next_state_tensor)
            log_probs = F.log_softmax(logits, dim=-1)
            log_q_z_given_s = log_probs[0, skill].item()
        
        # r(s,z) = log q(z|s') - log p(z)
        pseudo_reward = log_q_z_given_s - self.log_skill_prior
        
        return pseudo_reward
    
    def update_discriminator(self, batch_size=256):
        """Update discriminator to predict skill from next state"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        next_states = torch.FloatTensor(next_states).to(self.device)
        skills = torch.LongTensor(skills).to(self.device)
        
        # Compute discriminator logits
        logits = self.discriminator(next_states)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, skills)
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
        # Compute accuracy for logging
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == skills).float().mean().item()
        
        return loss.item(), accuracy
    
    def update_critic(self, batch_size=256):
        """Update critic networks using Bellman backup"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Create skill one-hot vectors
        skills_onehot = torch.zeros(batch_size, self.skill_dim).to(self.device)
        for i, skill in enumerate(skills):
            skills_onehot[i, skill] = 1.0
        
        # Compute target Q-value
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states, skills_onehot)
            target_q1 = self.critic1_target(next_states, next_actions, skills_onehot)
            target_q2 = self.critic2_target(next_states, next_actions, skills_onehot)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Update critic 1
        current_q1 = self.critic1(states, actions, skills_onehot)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        current_q2 = self.critic2(states, actions, skills_onehot)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def update_policy(self, batch_size=256):
        """Update policy network"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        
        # Create skill one-hot vectors
        skills_onehot = torch.zeros(batch_size, self.skill_dim).to(self.device)
        for i, skill in enumerate(skills):
            skills_onehot[i, skill] = 1.0
        
        # Sample actions from current policy
        new_actions, log_probs, _ = self.policy.sample(states, skills_onehot)
        
        # Compute Q-values
        q1 = self.critic1(states, new_actions, skills_onehot)
        q2 = self.critic2(states, new_actions, skills_onehot)
        q = torch.min(q1, q2)
        
        # Policy loss: maximize Q - alpha * log_prob
        policy_loss = (self.alpha * log_probs - q).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return policy_loss.item(), alpha_loss.item()
    
    def soft_update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(self, batch_size=256):
        """Perform one training step (update all networks)"""
        # Update discriminator
        disc_loss, disc_acc = self.update_discriminator(batch_size)
        
        # Update critics
        critic1_loss, critic2_loss = self.update_critic(batch_size)
        
        # Update policy
        policy_loss, alpha_loss = self.update_policy(batch_size)
        
        # Soft update target networks
        self.soft_update_target_networks()
        
        return {
            'discriminator_loss': disc_loss,
            'discriminator_accuracy': disc_acc,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }


# ==================== Training Loop ====================
def train_diayn(
    env_name='HalfCheetah-v5',
    num_episodes=1000,
    max_steps_per_episode=1000,
    num_skills=10,
    batch_size=256,
    updates_per_step=1,
    start_training_steps=1000,
    eval_interval=50,
    device='cpu',
    render_mode=None
):
    """
    Main training loop for DIAYN
    
    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        num_skills: Number of skills to discover
        batch_size: Batch size for training
        updates_per_step: Number of gradient updates per environment step
        start_training_steps: Start training after this many steps
        eval_interval: Evaluate skills every N episodes
        device: Device to use for training
        render_mode: Render mode for visualization (None, 'human', 'rgb_array')
    """
    # Create environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of skills: {num_skills}")
    
    # Initialize agent
    agent = DIAYN_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=num_skills,
        device=device
    )
    
    # Training statistics
    episode_rewards = []
    discriminator_accuracies = []
    total_steps = 0
    
    print("\nStarting training...")
    print("=" * 80)
    
    for episode in range(num_episodes):
        # Sample random skill for this episode
        skill = np.random.randint(0, num_skills)
        
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_pseudo_reward = 0
        episode_steps = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < start_training_steps:
                # Random actions for initial exploration
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, skill, deterministic=False)
            
            # Execute action
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Compute DIAYN pseudo-reward
            pseudo_reward = agent.compute_pseudo_reward(next_state, skill)
            
            # Store transition
            agent.replay_buffer.push(state, action, pseudo_reward, next_state, done, skill)
            
            # Update statistics
            episode_reward += env_reward  # External reward (for monitoring only)
            episode_pseudo_reward += pseudo_reward
            episode_steps += 1
            total_steps += 1
            
            # Update networks
            if total_steps >= start_training_steps:
                for _ in range(updates_per_step):
                    train_metrics = agent.train_step(batch_size)
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_pseudo_reward)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Skill: {skill} | "
                  f"Steps: {episode_steps} | "
                  f"Pseudo Reward: {episode_pseudo_reward:.2f} | "
                  f"Avg Reward (10 eps): {avg_reward:.2f} | "
                  f"Total Steps: {total_steps}")
            
            if total_steps >= start_training_steps:
                print(f"  Discriminator Loss: {train_metrics['discriminator_loss']:.4f} | "
                      f"Discriminator Acc: {train_metrics['discriminator_accuracy']:.4f} | "
                      f"Policy Loss: {train_metrics['policy_loss']:.4f} | "
                      f"Alpha: {train_metrics['alpha']:.4f}")
        
        # Evaluate skills
        if (episode + 1) % eval_interval == 0 and total_steps >= start_training_steps:
            print("\n" + "=" * 80)
            print(f"EVALUATION AT EPISODE {episode + 1}")
            print("=" * 80)
            evaluate_skills(env, agent, num_skills, num_episodes=3, max_steps=500)
            print("=" * 80 + "\n")
    
    env.close()
    return agent, episode_rewards


def evaluate_skills(env, agent, num_skills, num_episodes=5, max_steps=500):
    """
    Evaluate learned skills
    
    Args:
        env: Gymnasium environment
        agent: DIAYN_SAC agent
        num_skills: Number of skills
        num_episodes: Number of episodes per skill
        max_steps: Maximum steps per episode
    """
    print(f"Evaluating {num_skills} skills...")
    
    skill_rewards = []
    skill_positions = []
    
    for skill in range(num_skills):
        episode_rewards = []
        episode_positions = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            positions = [state[0]]  # Track x-position for HalfCheetah
            
            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                positions.append(next_state[0])
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_positions.append(positions[-1] - positions[0])  # Net displacement
        
        avg_reward = np.mean(episode_rewards)
        avg_displacement = np.mean(episode_positions)
        
        skill_rewards.append(avg_reward)
        skill_positions.append(avg_displacement)
        
        print(f"  Skill {skill}: Avg Reward: {avg_reward:.2f}, Avg Displacement: {avg_displacement:.2f}")
    
    # Print diversity metrics
    position_std = np.std(skill_positions)
    print(f"\nSkill Diversity (position std): {position_std:.2f}")
    
    return skill_rewards, skill_positions


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train DIAYN
    agent, rewards = train_diayn(
        env_name='HalfCheetah-v5',  # Changed to HalfCheetah for comparison
        num_episodes=1000,
        max_steps_per_episode=1000,
        num_skills=10,
        batch_size=256,
        updates_per_step=1,
        start_training_steps=1000,
        eval_interval=50,
        device=device,
        render_mode=None  # Disabled rendering for speed
    )
    
    print("\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")


