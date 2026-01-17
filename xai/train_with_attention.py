# Copyright 2025 Valeo.


"""Script to run training with attention weight logging for XAI analysis.

This script is similar to vmax/scripts/training/train.py but uses the XAI-enabled
PPO trainer that periodically logs attention weights for Head Specialization Analysis.

Usage:
    python xai/train_with_attention.py algorithm=PPO \\
        +xai.attention_log_freq=1000 \\
        +xai.attention_n_samples=4

The attention logs will be saved in the run directory under 'attention_logs/'.
"""

import os
import sys
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.scripts.training import train_utils

# Import XAI PPO trainer instead of standard one
from xai.xai_ppo_trainer import train as xai_train


OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir)


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    """Run the training process with attention logging.

    Args:
        cfg: Configuration for the training process.

    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    train_utils.apply_xla_flags(config)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()

    # Print XAI-specific info
    xai_config = config.get("xai", {})
    print(" XAI Configuration ".center(40, "="))
    print(f"- Attention Log Freq : {xai_config.get('attention_log_freq', 1000)}")
    print(f"- Attention N Samples: {xai_config.get('attention_n_samples', 4)}")

    env_config, run_config = train_utils.build_config_dicts(config)

    # (num_devices, num_envs, num_episode_per_epoch)
    data_generator = simulator.make_data_generator(
        path=env_config["path_dataset"],
        max_num_objects=env_config["max_num_objects"],
        include_sdc_paths=env_config["sdc_paths_from_data"],
        batch_dims=(env_config["num_envs"], env_config["num_episode_per_epoch"]),
        seed=env_config["seed"],
        distributed=True,
    )

    if config["eval_freq"] > 0:
        eval_data_generator = simulator.make_data_generator(
            path=env_config["path_dataset_eval"],
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            batch_dims=(8, config["num_scenario_per_eval"] // 8),
            seed=69,
            distributed=True,
        )
        eval_scenario = next(eval_data_generator)
        del eval_data_generator
    else:
        eval_scenario = None

    env = simulator.make_env_for_training(
        max_num_objects=env_config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=env_config["sdc_paths_from_data"],
        observation_type=env_config["observation_type"],
        observation_config=env_config["observation_config"],
        reward_type=env_config["reward_type"],
        reward_config=env_config["reward_config"],
        termination_keys=env_config["termination_keys"],
    )

    absolute_run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    relative_run_path = "/".join(absolute_run_path.split("/")[-2:])

    model_path = os.path.join(relative_run_path, "model")
    os.makedirs(model_path, exist_ok=True)

    writer = train_utils.setup_tensorboard(relative_run_path)
    progress = partial(train_utils.log_metrics, writer=writer)

    # XAI-specific configuration
    attention_log_dir = absolute_run_path  # Attention logs go in run directory
    attention_log_freq = xai_config.get("attention_log_freq", 1000)
    attention_n_samples = xai_config.get("attention_n_samples", 4)

    ## TRAINING WITH ATTENTION LOGGING
    # Ensure we're using PPO (only algorithm supported for attention logging)
    if config["algorithm"]["name"] != "PPO":
        print(f"[WARNING] XAI training only supports PPO, got {config['algorithm']['name']}")
        print("[WARNING] Falling back to standard training without attention logging")
        from vmax.agents import learning
        train_fn = learning.get_train_fn(config["algorithm"]["name"])
        train_fn(
            env=env,
            data_generator=data_generator,
            eval_scenario=eval_scenario,
            **run_config,
            progress_fn=progress,
            checkpoint_logdir=model_path,
            disable_tqdm=not sys.stdout.isatty(),
        )
    else:
        # Use XAI-enabled PPO trainer
        xai_train(
            env=env,
            data_generator=data_generator,
            eval_scenario=eval_scenario,
            **run_config,
            progress_fn=progress,
            checkpoint_logdir=model_path,
            disable_tqdm=not sys.stdout.isatty(),
            # XAI-specific parameters
            attention_log_freq=attention_log_freq,
            attention_log_dir=attention_log_dir,
            attention_n_samples=attention_n_samples,
        )


if __name__ == "__main__":
    run()
