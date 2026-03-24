import torch
from torch import nn
import numpy as np


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 64), std=np.sqrt(2)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64), std=np.sqrt(2)),
            nn.Tanh()
        )

        self.mean = init_layer(nn.Linear(64, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean) 
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, 64), std=np.sqrt(2)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64), std=np.sqrt(2)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, x):
        return self.net(x)