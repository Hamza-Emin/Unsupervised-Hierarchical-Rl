"""
Record videos of DIAYN skills
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import os
from dıayn_sac import DIAYN_SAC


def load_agent(checkpoint_path, state_dim, action_dim, skill_dim, hidden_dim=128, device='cpu'):
    agent = DIAYN_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_dim=hidden_dim,
        device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    agent.policy.eval()
    return agent


def record_skill_video(agent, env_name, skill, output_dir="videos", max_steps=1000, video_name=None):
    """Record a video of a single skill"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with video recording
    if video_name is None:
        video_name = f"skill_{skill}"
    
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_folder=output_dir,
        name_prefix=video_name,
        episode_trigger=lambda x: True  # Record every episode
    )
    
    # Run episode
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.select_action(state, skill, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
    
    env.close()
    print(f"✅ Skill {skill} video saved to {output_dir}/ (Reward: {total_reward:.2f})")
    return total_reward


def record_all_skills(checkpoint_path, env_name, num_skills=10, hidden_dim=128, 
                      output_dir="videos", max_steps=1000):
    """Record videos of all skills"""
    
    # Get env dimensions
    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    # Load agent
    print(f"Loading agent from {checkpoint_path}...")
    agent = load_agent(checkpoint_path, state_dim, action_dim, num_skills, hidden_dim)
    print("✅ Agent loaded!")
    
    print(f"\n📹 Recording {num_skills} skill videos to '{output_dir}/'...\n")
    
    for skill in range(num_skills):
        record_skill_video(agent, env_name, skill, output_dir, max_steps)
    
    print(f"\n🎉 All videos saved to '{output_dir}/' folder!")


if __name__ == "__main__":
    # === CONFIGURE HERE ===
    
    # For HalfCheetah agent (diayn_agent3)
    
    
    
    
    # # For Ant agent (diayn_agent1) with hidden_dim=256
    record_all_skills(
         checkpoint_path=r"C:\Users\hamza\Masaüstü\Rl_project\diayn_agent.pth",
         env_name="Ant-v5",
         num_skills=10,
         hidden_dim=256,
         output_dir="videos_ant_256",
         max_steps=1000
     )