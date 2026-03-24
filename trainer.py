import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from collections import deque
import wandb
from abc import ABC, abstractmethod
from torch import nn

from actor_critic import Actor, Critic
from rl_utils import compute_gae


class Trainer(ABC):
    def __init__(self, envs, config):
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.obs_dim = envs.single_observation_space.shape[0]
        self.act_dim = envs.single_action_space.shape[0]

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        
        optimizer_name = getattr(self.config, "optim", "adam")
        if optimizer_name.lower() == "rmsprop":
            self.policy_optimizer = optim.RMSprop(
                self.actor.parameters(),
                lr=getattr(self.config, "policy_lr", 1e-3),
                eps=getattr(self.config, "optim_eps", 1e-5),
                alpha=getattr(self.config, "optim_alpha", 0.99),
            )
            self.value_optimizer = optim.RMSprop(
                self.critic.parameters(),
                lr=getattr(self.config, "value_lr", 1e-3),
                eps=getattr(self.config, "optim_eps", 1e-5),
                alpha=getattr(self.config, "optim_alpha", 0.99),
            )
        elif optimizer_name.lower() == "adam":
            self.policy_optimizer = optim.Adam(
                self.actor.parameters(),
                lr=getattr(self.config, "policy_lr", 1e-3),
                eps=getattr(self.config, "optim_eps", 1e-5),
            )
            self.value_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=getattr(self.config, "value_lr", 1e-3),
                eps=getattr(self.config, "optim_eps", 1e-5),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def save_checkpoint(self, step=None):
        save_path = self.config.save_path
        if step is not None:
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_{step}{ext}"

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        config_dict = dict(self.config) if getattr(self.config, 'use_wandb', False) else vars(self.config)
        
        try:
            obs_rms = self.envs.envs[0].env.env.env.obs_rms
            obs_rms_mean = obs_rms.mean
            obs_rms_var = obs_rms.var
        except AttributeError:
            obs_rms_mean, obs_rms_var = None, None
            print("WARNING: Could not find obs_rms at the expected wrapper depth.")

        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "config": config_dict,
            "obs_rms_mean": obs_rms_mean, 
            "obs_rms_var": obs_rms_var
        }
        
        torch.save(checkpoint, save_path)
        print(f"\n*** Checkpoint saved: {save_path} ***\n")

    def linear_lr_schedule(self, step, total_steps, initial_lr, optimizer):
        """Linearly anneals learning rate down to 0."""
        frac = 1.0 - (step - 1.0) / total_steps
        lr = frac * initial_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @torch.no_grad()
    def rollout(self, obs, recent_returns, global_step):
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        for _ in range(self.config.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            states.append(obs_tensor)
            
            mean, std = self.actor(obs_tensor)
            dist = D.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            
            # 1. Treat truncations as 'done' so GAE cuts off advantage flow from the new reset state.
            done = np.logical_or(terminated, truncated)
            
            # 2. If truncated, manually bootstrap the true final state and inject it into the reward.
            if "_final_observation" in infos:
                for idx, has_final_obs in enumerate(infos["_final_observation"]):
                    if has_final_obs and truncated[idx]:
                        final_obs = infos["final_observation"][idx]
                        # Add batch dimension for the Value Network
                        final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        
                        terminal_value = self.critic(final_obs_tensor).squeeze(-1).item()
                        
                        # Inject the bootstrapped value into the reward
                        reward[idx] += self.config.gamma * terminal_value

            dones.append(torch.tensor(done, dtype=torch.float32, device=self.device).view(-1))

            if "episode" in infos and "_episode" in infos:
                mask = infos["_episode"]
                episode_returns = infos["episode"]["r"][mask]
                for r in episode_returns:
                    recent_returns.append(r.item())

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1))
            
            obs = next_obs
            global_step += self.config.num_envs

        return obs, states, actions, rewards, dones, log_probs, global_step, recent_returns

    def train(self):
        obs, _ = self.envs.reset(seed=self.config.seed)
        global_step = 0
        update_step = 0
        recent_returns = deque(maxlen=100)
        
        save_interval = getattr(self.config, 'save_interval', 1000000)
        next_save_step = save_interval

        while global_step < self.config.total_steps:
            # Anneal learning rates
            if getattr(self.config, 'anneal_lr', True):
                self.linear_lr_schedule(global_step, self.config.total_steps, self.config.policy_lr, self.policy_optimizer)
                self.linear_lr_schedule(global_step, self.config.total_steps, self.config.value_lr, self.value_optimizer)

            obs, states, actions, rewards, dones, log_probs, global_step, recent_returns = self.rollout(
                obs, recent_returns, global_step
            )

            metrics = self.update(states, actions, rewards, dones, log_probs, obs)
            update_step += 1
            metrics["train/policy_lr"] = self.policy_optimizer.param_groups[0]["lr"]
            metrics["train/value_lr"] = self.value_optimizer.param_groups[0]["lr"]
            
            if len(recent_returns) > 0:
                smoothed_return = np.mean(recent_returns)
                metrics["train/smoothed_return"] = smoothed_return
                metrics["global_step"] = global_step

                if getattr(self.config, 'use_wandb', False):
                    wandb.log(metrics, step=global_step)
                    
                print(f"Update: {update_step} | Step: {global_step} | Return: {smoothed_return:.2f}")

            if global_step >= next_save_step:
                self.save_checkpoint(step=global_step)
                next_save_step += save_interval

        self.save_checkpoint(step="final")

    @abstractmethod
    def update(self, states, actions, rewards, dones, log_probs, next_obs):
        pass


class VPGTrainer(Trainer):
    def update(self, states, actions, rewards, dones, old_log_probs, next_obs):
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)

        with torch.no_grad():
            values = self.critic(states_tensor).squeeze(-1)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_value = self.critic(next_obs_tensor).squeeze(-1)
            
        gae_lambda = getattr(self.config, 'gae_lambda', 0.97)
        advantages, returns = compute_gae(
            rewards, values, dones, next_value, self.config.gamma, gae_lambda
        )

        y_pred, y_true = values.detach().cpu().numpy(), returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean, std = self.actor(states_tensor)
        dist = D.Normal(mean, std)
        log_probs_tensor = dist.log_prob(actions_tensor).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        ent_coef = getattr(self.config, 'ent_coef', 0.0)
        policy_loss = -(log_probs_tensor * advantages).mean() - (ent_coef * entropy)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        a_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5).item()
        self.policy_optimizer.step()

        value_epochs = getattr(self.config, 'value_epochs', 10)
        v_loss_val = 0
        for _ in range(value_epochs):
            v_preds = self.critic(states_tensor).squeeze(-1)
            value_loss = F.mse_loss(v_preds, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.value_optimizer.step()
            v_loss_val = value_loss.item()
        
        return {
            "train/policy_loss": policy_loss.item(), 
            "train/value_loss": v_loss_val,
            "train/entropy": entropy.item(),
            "train/explained_variance": explained_var,
            "train/actor_grad_norm": a_grad_norm
        }


class A2CTrainer(Trainer):
    def update(self, states, actions, rewards, dones, old_log_probs, next_obs):
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        
        values = self.critic(states_tensor).squeeze(-1)
        
        with torch.no_grad():
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_value = self.critic(next_obs_tensor).squeeze(-1)
            
        gae_lambda = getattr(self.config, 'gae_lambda', 0.95)
        advantages, returns = compute_gae(
            rewards, values.detach(), dones, next_value, self.config.gamma, gae_lambda
        )
        
        y_pred, y_true = values.detach().cpu().numpy(), returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        mean, std = self.actor(states_tensor)
        dist = D.Normal(mean, std)
        log_probs_tensor = dist.log_prob(actions_tensor).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        ent_coef = getattr(self.config, 'ent_coef', 0.0)
        policy_loss = -(log_probs_tensor * advantages).mean() - (ent_coef * entropy)
        value_loss = F.mse_loss(values, returns)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        a_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5).item()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.value_optimizer.step()
        
        return {
            "train/policy_loss": policy_loss.item(), 
            "train/value_loss": value_loss.item(),
            "train/entropy": entropy.item(),
            "train/explained_variance": explained_var,
            "train/actor_grad_norm": a_grad_norm
        }
    

class PPOTrainer(Trainer):
    def update(self, states, actions, rewards, dones, old_log_probs, next_obs):
        states_seq = torch.stack(states).detach()
        with torch.no_grad():
            values = self.critic(states_seq).squeeze(-1)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_value = self.critic(next_obs_tensor).squeeze(-1)
            
        gae_lambda = getattr(self.config, 'gae_lambda', 0.95)
        advantages, returns = compute_gae(
            rewards, values, dones, next_value, self.config.gamma, gae_lambda
        )

        y_pred, y_true = values.detach().cpu().numpy(), returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        b_states = torch.stack(states).detach().reshape(-1, self.obs_dim)
        b_actions = torch.stack(actions).detach().reshape(-1, self.act_dim)
        b_old_log_probs = torch.stack(old_log_probs).detach().reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        batch_size = b_states.shape[0]
        minibatch_size = getattr(self.config, 'minibatch_size', 64)
        update_epochs = getattr(self.config, 'update_epochs', 10)
        clip_coef = getattr(self.config, 'clip_eps', 0.2)
        ent_coef = getattr(self.config, 'ent_coef', 0.0)
        vf_coef = getattr(self.config, 'vf_coef', 0.5)
        max_grad_norm = getattr(self.config, 'max_grad_norm', 0.5)
        
        b_inds = np.arange(batch_size)
        total_policy_loss, total_value_loss, total_entropy = 0, 0, 0
        total_kl, total_clip_frac = 0, 0
        
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                mean, std = self.actor(b_states[mb_inds])
                dist = D.Normal(mean, std)
                new_log_probs = dist.log_prob(b_actions[mb_inds]).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()
                
                log_ratio = new_log_probs - b_old_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio < 1 - clip_coef) | (ratio > 1 + clip_coef)).float().mean().item()
                
                surr1 = ratio * b_advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_advantages[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                mb_values = self.critic(b_states[mb_inds]).squeeze(-1)
                value_loss = F.mse_loss(mb_values, b_returns[mb_inds])
                
                loss = policy_loss - (ent_coef * entropy) + (value_loss * vf_coef)
                
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl
                total_clip_frac += clip_frac
                
        num_updates = update_epochs * (batch_size // minibatch_size)
        
        return {
            "train/policy_loss": total_policy_loss / num_updates,
            "train/value_loss": total_value_loss / num_updates,
            "train/entropy": total_entropy / num_updates,
            "train/explained_variance": explained_var,
            "train/approx_kl": total_kl / num_updates,
            "train/clip_fraction": total_clip_frac / num_updates
        }