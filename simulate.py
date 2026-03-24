import os
import time
import argparse
import torch
import gymnasium as gym
import numpy as np

from actor_critic import Actor 


def main():
    parser = argparse.ArgumentParser(description="Record videos of a trained MuJoCo agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.pt file")
    parser.add_argument("--save-dir", type=str, default="videos", help="Directory to save the mp4 videos")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to simulate and record")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    env_name = config["env_name"] if isinstance(config, dict) else config.env_name
    print(f"Environment found in config: {env_name}")

    os.makedirs(args.save_dir, exist_ok=True)
    
    run_id = int(time.time())
    unique_video_dir = os.path.join(args.save_dir, f"{config["algorithm"]}_{env_name}_{run_id}")
    os.makedirs(unique_video_dir, exist_ok=True)
    
    base_env = gym.make(env_name, render_mode="rgb_array")
    
    env = gym.wrappers.RecordVideo(
        base_env, 
        video_folder=unique_video_dir,
        episode_trigger=lambda x: True, 
        name_prefix=f"{env_name}-eval"
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    obs_mean = checkpoint.get("obs_rms_mean")
    obs_var = checkpoint.get("obs_rms_var")

    if obs_mean is not None and obs_var is not None:
        print("Loaded normalization statistics successfully!")
    else:
        print("WARNING: No normalization stats found! Defaulting to raw observations.")
        obs_mean = np.zeros(obs_dim)
        obs_var = np.ones(obs_dim)

    actor = Actor(obs_dim, act_dim).to(device)
    
    state_dict_key = "actor_state_dict"
    actor.load_state_dict(checkpoint[state_dict_key])

    actor.eval()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0

        while not done:
            norm_obs = (obs - obs_mean) / np.sqrt(obs_var + 1e-8)
            norm_obs = np.clip(norm_obs, -10.0, 10.0)

            with torch.no_grad():
                obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                mean, std = actor(obs_tensor)
                action = mean.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}/{args.episodes} completed. Total Reward: {episode_reward:.2f}")

    env.close()
    print(f"\nVideos saved successfully to: {os.path.abspath(unique_video_dir)}")

if __name__ == "__main__":
    main()