import gymnasium as gym
import numpy as np


def make_env(env_name, seed, gamma=0.99):
    def thunk():
        env = gym.make(env_name)
        # Record stats before any normalization alters the raw score
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # env = gym.wrappers.ClipAction(env)
        
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space
        )
        
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def create_vector_env(env_name, num_envs, seed, gamma=0.99):
    env_fns = [make_env(env_name, seed + i, gamma) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)