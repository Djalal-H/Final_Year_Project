import os
import argparse
import jax
import numpy as np
import yaml
from functools import partial
from tqdm import tqdm
from waymax import dynamics

from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator, make_env_for_evaluation
from vmax.agents import pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Quick evaluation script for pretrained models")
    parser.add_argument("--model_id", "-m", type=str, required=True, help="Identifier of the model to evaluate (e.g., 'ppo_training_2024...')")
    parser.add_argument("--num_scenarios", "-n", type=int, default=10, help="Number of scenarios to evaluate (default: 10)")
    parser.add_argument("--dataset_path", "-d", type=str, default="local_womd_valid", help="Path or name of the dataset to use (default: local_womd_valid)")
    parser.add_argument("--src_dir", "-s", type=str, default="runs", help="Source directory for model checkpoints (default: runs)")
    parser.add_argument("--max_num_objects", "-o", type=int, default=64, help="Maximum number of objects in the scene (default: 64)")
    
    return parser.parse_args()

def print_model_info(config_path):
    print("\n" + "="*50)
    print("      PRETRAINED MODEL CONFIGURATION INFO")
    print("="*50)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract important parameters
        algo_name = config.get("algorithm", {}).get("name", "N/A")
        encoder_type = config.get("network", {}).get("encoder", {}).get("_target_", "N/A").split('.')[-1]
        policy_type = config.get("algorithm", {}).get("network", {}).get("policy", {}).get("_target_", "N/A").split('.')[-1]
        obs_type = config.get("observation_type", "N/A")
        term_keys = config.get("termination_keys", [])
        learning_rate = config.get("algorithm", {}).get("learning_rate", "N/A")
        
        print(f"{'Algorithm:':<20} {algo_name}")
        print(f"{'Encoder:':<20} {encoder_type}")
        print(f"{'Policy:':<20} {policy_type}")
        print(f"{'Observation:':<20} {obs_type}")
        print(f"{'Termination:':<20} {', '.join(term_keys)}")
        print(f"{'Learning Rate:':<20} {learning_rate}")
        print("="*50 + "\n")
        
        return config
    except Exception as e:
        print(f"Error reading config from {config_path}: {e}")
        return None

def main():
    args = parse_args()
    
    model_dir = os.path.join(args.src_dir, args.model_id)
    config_path = os.path.join(model_dir, ".hydra/config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    # 1. Print Model Info
    config = print_model_info(config_path)
    if config is None:
        return

    # 2. Setup Evaluation
    print(f"-> Setting up evaluation for model: {args.model_id}")
    
    # Prepare config for evaluation setup (mimicking utils.setup_evaluation)
    config["encoder"] = config["network"]["encoder"]
    config["policy"] = config["algorithm"]["network"]["policy"]
    config["value"] = config["algorithm"]["network"]["value"] if "value" in config["algorithm"]["network"] else None
    config["unflatten_config"] = config["observation_config"]
    config["action_distribution"] = config["algorithm"]["network"]["action_distribution"]
    
    termination_keys = config["termination_keys"]
    
    # Create environment
    env = make_env_for_evaluation(
        max_num_objects=args.max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=config["observation_type"],
        observation_config=config["observation_config"],
        termination_keys=termination_keys,
        noisy_init=False,
    )
    
    # Identify checkpoint
    model_path_dir = os.path.join(model_dir, "model/")
    try:
        model_path, model_name = utils.get_model_path(model_path_dir)
    except Exception as e:
        print(f"Error finding model checkpoint: {e}")
        return

    # Load policy
    print(f"-> Loading model from: {model_path}")
    policy = utils.load_model(env, config["algorithm"]["name"], config, model_path)
    step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy)
    
    # 3. Run Quick Evaluation
    print(f"-> Running evaluation on {args.num_scenarios} scenarios...")
    
    data_generator = make_data_generator(
        path=datasets.get_dataset(args.dataset_path),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),  # Sequential for quick eval
        seed=0,
        repeat=1,
    )
    
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)
    
    rng_key = jax.random.PRNGKey(0)
    eval_metrics = {"episode_length": [], "accuracy": []}
    
    for i, scenario in enumerate(tqdm(data_generator, total=args.num_scenarios, desc="Evaluating")):
        if i >= args.num_scenarios:
            break
            
        rng_key, scenario_key = jax.random.split(rng_key)
        
        # Run scenario (simplified run_scenario_jit for single scenario)
        scenario_key = jax.random.split(scenario_key, 1)
        episode_metrics, steps_done = utils.run_scenario_jit(
            scenario, 
            scenario_key, 
            step_fn=jitted_step_fn, 
            reset_fn=jitted_reset
        )
        
        # Aggregate metrics
        eval_metrics = utils.append_episode_metrics(
            steps_done,
            eval_metrics,
            episode_metrics,
            termination_keys,
            batch_size=1
        )

    # 4. Print Results
    print("\n" + "="*50)
    print("            EVALUATION SUMMARY")
    print("="*50)
    print(f"{'Scenarios:':<20} {len(eval_metrics['accuracy'])}")
    print(f"{'Mean Accuracy:':<20} {np.mean(eval_metrics['accuracy']):.4f}")
    if "vmax_aggregate_score" in eval_metrics:
        print(f"{'V-Max Score:':<20} {np.mean(eval_metrics['vmax_aggregate_score']):.4f}")
    if "episode_length" in eval_metrics:
        print(f"{'Avg Episode Len:':<20} {np.mean(eval_metrics['episode_length']):.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
