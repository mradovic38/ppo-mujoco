import torch
import numpy as np
import random


def compute_returns(rewards, dones, gamma):
    returns = []
    G = torch.zeros_like(rewards[0])

    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - d)
        returns.insert(0, G)

    returns = torch.stack(returns)
    return returns


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = []
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages.insert(0, last_gae_lam)
        
    advantages = torch.stack(advantages)
    
    # In GAE, Returns = Advantages + Values
    returns = advantages + values 
    return advantages, returns


def set_seed(seed, envs=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if envs is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)