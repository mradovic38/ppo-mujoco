# Proximal Policy Optimization (PPO) for MuJoCo
This repository contains an implementation of **Proximal Policy Optimization (PPO)** based on the original PPO paper for continuous control tasks in **MuJoCo** environments. The implementation has been tested on **HalfCheetah**, **Swimmer**, **Hopper**, and **Walker2d** and compared with **A2C** and **Vanilla Policy Gradient (VPG)**.


<div style="display: flex; flex-wrap: wrap; gap: 1em;">

  <div style="flex: 0 0 22%;">
    <video src="assets/video_examples/ppo_HalfCheetah-v5_1774206214/HalfCheetah-v5-eval-episode-0.mp4" style="width:100%;" autoplay loop muted></video>
    <p style="text-align:center; font-size:0.9em;">HalfCheetah-v5</p>
  </div>

  <div style="flex: 0 0 22%;">
    <video src="assets/video_examples/ppo_Hopper-v5_1774206517/Hopper-v5-eval-episode-0.mp4" style="width:100%;" autoplay loop muted></video>
    <p style="text-align:center; font-size:0.9em;">Hopper-v5</p>
  </div>

  <div style="flex: 0 0 22%;">
    <video src="assets/video_examples/ppo_Swimmer-v5_1774206403/Swimmer-v5-eval-episode-0.mp4" style="width:100%;" autoplay loop muted></video>
    <p style="text-align:center; font-size:0.9em;">Swimmer-v5</p>
  </div>

  <div style="flex: 0 0 22%;">
    <video src="assets/video_examples/ppo_Walker2d-v5_1774206598/Walker2d-v5-eval-episode-0.mp4" style="width:100%;" autoplay loop muted></video>
    <p style="text-align:center; font-size:0.9em;">Walker2d-v5</p>
  </div>

</div>


## Prerequisites
1) Install UV (if you don't have it already)
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Install dependencies
```sh
uv sync
```

3) If you want to use W&B to track training progress: Generate [W&B](https://wandb.ai/) API key, create .env file and add `WANDB_API_KEY`:
```
WANDB_API_KEY=<YOUR_API_KEY>
```

## Training
### Regular training with W&B
```sh
uv run python train.py --config config/<MUJOCO_ENV_NAME>/<a2c|ppo|vpg>.yaml
```

### Regular training without W&B
```sh
uv run python train.py --config config/<MUJOCO_ENV_NAME>/<a2c|ppo|vpg>.yaml --disable-wandb
```

### W&B sweep (runs all 3 random seeds)
```sh
wandb sweep config/<MUJOCO_ENV_NAME>/<a2c|ppo|vpg>.yaml      
wandb agent <AGENT_NAME>
```   


## Simulating
```sh
uv run python simulate.py --config config/<MUJOCO_ENV_NAME>/<a2c|ppo|vpg>.yaml --video-dir videos --episodes <NUM_EPISODES>
```

## Supported Environments
* [HalfCheetah-v5](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
* [Swimmer-v5](https://gymnasium.farama.org/environments/mujoco/swimmer/)
* [Hopper-v5](https://gymnasium.farama.org/environments/mujoco/hopper/)
* [Walker2d-v5](https://gymnasium.farama.org/environments/mujoco/walker2d/)

## Training Performance

I log **smoothed returns over the last 100 episodes** during training. Below are the learning curves of **PPO**, **A2C**, and **Vanilla PG**, averaged over 3 random seeds.

<div style="display: flex; flex-wrap: wrap; gap: 1em;">

  <div style="flex: 0 0 22%;">
    <img src="assets/figures/HalfCheetah.png" alt="HalfCheetah Training" style="width:100%">
  </div>

  <div style="flex: 0 0 22%;">
    <img src="assets/figures/Hopper.png" alt="Hopper Training" style="width:100%">
  </div>

  <div style="flex: 0 0 22%;">
    <img src="assets/figures/Swimmer.png" alt="Swimmer Training" style="width:100%">
  </div>

  <div style="flex: 0 0 22%;">
    <img src="assets/figures/Walker2d.png" alt="Walker2d Training" style="width:100%">
  </div>

</div>
<p><strong>Legend:</strong> <span style="color:#1f77b4;">PPO</span> | <span style="color:#ff7f0e;">A2C</span> | <span style="color:#2ca02c;">Vanilla PG</span></p>

<p style="font-size:0.9em;">Comparison of PPO, A2C, and Vanilla PG algorithms on different MuJoCo environments: average return over 100 episodes, trained for 1 million timesteps.</p>

## Resources
* [Proximal Policy Optimization Algorithms, Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
* [Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
* [MuJoCo Gymnasium Documentation](https://gymnasium.farama.org/environments/mujoco/)