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
            
        # Helper to find a value in nested dict
        def find_nested(d, keys, default="N/A"):
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return default
            return d

        algo_name = config.get("algorithm", {}).get("name", config.get("name", "N/A"))
        
        # Try different ways to find encoder/policy types
        encoder_raw = config.get("network", {}).get("encoder", {})
        if isinstance(encoder_raw, dict):
            encoder_type = encoder_raw.get("_target_", "N/A").split('.')[-1]
            if encoder_type == "N/A":
                encoder_type = encoder_raw.get("type", "N/A")
        else:
            encoder_type = str(encoder_raw)

        policy_raw = config.get("algorithm", {}).get("network", {}).get("policy", {})
        if isinstance(policy_raw, dict):
            policy_type = policy_raw.get("_target_", "N/A").split('.')[-1]
            if policy_type == "N/A":
                policy_type = policy_raw.get("type", "MLP") # Default to MLP for SAC/BC
        else:
            policy_type = str(policy_raw)

        obs_type = config.get("observation_type", "N/A")
        term_keys = config.get("termination_keys", [])
        learning_rate = config.get("algorithm", {}).get("learning_rate", config.get("learning_rate", "N/A"))
        
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
    
    # Flatten nested config for compatibility
    # Try multiple locations for these fields as Hydra configs varies
    if "algorithm" in config and "network" in config["algorithm"]:
        nw = config["algorithm"]["network"]
        config["policy"] = nw.get("policy", config.get("policy", {}))
        config["value"] = nw.get("value", config.get("value", {}))
        
        # Robust action_distribution extraction
        ad = nw.get("action_distribution", config.get("action_distribution", "gaussian"))
        if isinstance(ad, dict):
            # If it's a dict, try to find a string that looks like 'gaussian' or 'beta'
            ad_str = str(ad).lower()
            if "beta" in ad_str: config["action_distribution"] = "beta"
            else: config["action_distribution"] = "gaussian"
        else:
            config["action_distribution"] = ad
    
    if "network" in config and "encoder" in config["network"]:
        config["encoder"] = config["network"]["encoder"]
    elif "encoder" not in config:
        config["encoder"] = {"_target_": "vmax.agents.networks.encoders.MLPEncoder"}

    config["unflatten_config"] = config.get("observation_config", {})
    
    # Check if config has unsupported features - we'll need to handle them
    unsupported_features = []
    if "roadgraphs" in config["observation_config"] and "features" in config["observation_config"]["roadgraphs"]:
        SUPPORTED_ROADGRAPH_FEATURES = ["waypoints", "direction", "valid", "types"]
        original_features = config["observation_config"]["roadgraphs"]["features"]
        unsupported_features = [f for f in original_features if f not in SUPPORTED_ROADGRAPH_FEATURES]
        
        if unsupported_features:
            print(f"Warning: Model was trained with unsupported roadgraph features: {unsupported_features}")
            print(f"These will be replaced with zeros during evaluation.")
            # Remove them from config so observation extraction doesn't fail
            config["observation_config"]["roadgraphs"]["features"] = [
                f for f in original_features if f in SUPPORTED_ROADGRAPH_FEATURES
            ]
    
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
    
    # Create a wrapper to pad observations if needed
    def pad_observation_for_missing_features(obs, num_missing_rg_features=0):
        """Pad roadgraph features with zeros if model expects more features than provided."""
        if num_missing_rg_features == 0:
            return obs
        
        # Calculate where roadgraph features are in the flattened observation
        object_features_size = env.get_wrapper_attr("features_extractor").get_features_size(
            env.get_wrapper_attr("features_extractor")._object_features_key
        )
        sdc_size = 1 * args.obs_past_num_steps * object_features_size if hasattr(args, 'obs_past_num_steps') else 1 * 5 * object_features_size
        other_size = env.get_wrapper_attr("features_extractor")._num_closest_objects * \
                     env.get_wrapper_attr("features_extractor")._obs_past_num_steps * object_features_size
        
        rg_start = sdc_size + other_size
        rg_points = env.get_wrapper_attr("features_extractor")._roadgraph_top_k
        rg_features_current = env.get_wrapper_attr("features_extractor").get_features_size(
            env.get_wrapper_attr("features_extractor")._roadgraph_features_key
        )
        rg_size_current = rg_points * rg_features_current
        rg_end = rg_start + rg_size_current
        
        # Split observation into parts
        before_rg = obs[..., :rg_start]
        rg_obs = obs[..., rg_start:rg_end]
        after_rg = obs[..., rg_end:]
        
        # Reshape roadgraph to (points, features)
        rg_reshaped = rg_obs.reshape(*obs.shape[:-1], rg_points, rg_features_current)
        
        # Add zero padding for missing features (insert before 'valid' which is last)
        # Current: [xy(2), dir_xy(2), valid(1)] = 5 features
        # Need: [xy(2), dir_xy(2), speed_limit(1), valid(1)] = 6 features  
        # Insert zeros at position -1 (before valid)
        padding = jnp.zeros((*rg_reshaped.shape[:-1], num_missing_rg_features))
        rg_padded = jnp.concatenate([
            rg_reshaped[..., :-1],  # All except valid
            padding,                 # Missing features as zeros
            rg_reshaped[..., -1:]   # Valid flag
        ], axis=-1)
        
        # Flatten back
        rg_padded_flat = rg_padded.reshape(*obs.shape[:-1], -1)
        
        # Reconstruct observation
        return jnp.concatenate([before_rg, rg_padded_flat, after_rg], axis=-1)
    
    # Determine if we need padding
    num_missing = len(unsupported_features) if unsupported_features else 0
    
    # Wrap the step function to apply padding
    original_step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy)
    
    if num_missing > 0:
        def padded_step_fn(env_transition, key):
            # Pad the observation before passing to policy
            padded_obs = pad_observation_for_missing_features(env_transition.observation, num_missing)
            padded_transition = env_transition.replace(observation=padded_obs)
            return original_step_fn(padded_transition, key=key)
        step_fn = padded_step_fn
    else:
        step_fn = original_step_fn

    
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
        
        # Run scenario - run_scenario_jit expects a single key, not batched
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
