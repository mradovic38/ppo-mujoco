import os
import argparse
import yaml
import time
import wandb
from dotenv import load_dotenv
from types import SimpleNamespace

from env import create_vector_env
from rl_utils import set_seed
from trainer import A2CTrainer, PPOTrainer, VPGTrainer

TRAINER_REGISTRY = {
    "vpg": VPGTrainer,
    "a2c": A2CTrainer,
    "ppo": PPOTrainer
}


def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"Note: Config file {yaml_path} not found. Using CLI/Sweep arguments.")
        return {}, "ppo-mujoco"

    with open(yaml_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config_dict = {}
    if 'parameters' in raw_config:
        for key, val in raw_config['parameters'].items():
            clean_key = key.replace('-', '_')
            if 'value' in val:
                config_dict[clean_key] = val['value']
            elif 'values' in val:
                config_dict[clean_key] = val['values'][0] 

    project = raw_config.get('project', 'ppo-mujoco')
    return config_dict, project

def main():
    start = time.time()
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--disable-wandb", action="store_true")
    
    args, unknown = parser.parse_known_args()

    default_config, project = load_config(args.config)

    for i in range(len(unknown)):
        arg = unknown[i]
        if arg.startswith("--"):
            if "=" in arg:
                key_raw, val = arg.lstrip("-").split("=")
            else:
                key_raw = arg.lstrip("-")
                val = unknown[i+1] if i+1 < len(unknown) else True

            if key_raw == "env-name":
                key = "env_name" 
            else:
                key = key_raw.replace("-", "_")

            try:
                if isinstance(val, str):
                    if "." in val: val = float(val)
                    else: val = int(val)
            except ValueError:
                pass
            
            default_config[key] = val

    if args.disable_wandb:
        print("W&B disabled. Running locally.")
        config = SimpleNamespace(**default_config)
        # Manually append seed to save path for local runs
        base, ext = os.path.splitext(config.save_path)
        config.save_path = f"{base}_seed{config.seed}{ext}"
        config.use_wandb = False
    else:
        wandb.init(project=project, config=default_config)
        config = wandb.config
        
        algo_str = str(config.algorithm).upper()
        wandb.run.name = f"{algo_str}-{config.env_name}-seed{config.seed}"
        
        base, ext = os.path.splitext(config.save_path)
        new_save_path = f"{base}_seed{config.seed}{ext}"
        config.update({"save_path": new_save_path, "use_wandb": True}, allow_val_change=True)
        
    set_seed(config.seed)
    envs = create_vector_env(
        config.env_name,
        config.num_envs,
        config.seed,
        gamma=config.gamma,
    )

    algo_name = config.algorithm.lower()
    if algo_name not in TRAINER_REGISTRY:
        raise ValueError(f"Algorithm '{algo_name}' is not registered. Available: {list(TRAINER_REGISTRY.keys())}")
    
    TrainerClass = TRAINER_REGISTRY[algo_name]
    trainer = TrainerClass(envs, config)
    
    trainer.train()

    envs.close()
    if getattr(config, 'use_wandb', False):
        wandb.finish()
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()