# %% [markdown]
# # Offline Attention Extraction
# 
# This notebook loads a trained model and extracts attention weights from the Wayformer encoder
# by running inference on scenarios from the dataset.
# 
# **Workflow:**
# 1. Load trained model checkpoint
# 2. Load scenarios from dataset
# 3. Run forward pass with `return_attention_weights=True`
# 4. Save/analyze the extracted attention weights

# %%
# Setup
import os
import sys
import pickle
import glob
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, datasets, make_data_generator
from vmax.agents.learning.reinforcement.ppo import ppo_factory
from vmax.scripts.evaluate.utils import load_params, load_yaml_config

print("✓ Libraries loaded")

# %% [markdown]
# ## 1. Configuration
# Set paths to your trained model and dataset.

# %%
# === CONFIGURE THESE ===
RUN_DIR = "runs/PPO_VEC_WAYFORMER_your_run_name/"  # Path to training run
DATASET_PATH = "training"  # Dataset name or path
NUM_SCENARIOS = 10  # Number of scenarios to process
OUTPUT_DIR = "attention_extractions/"  # Where to save results

# Derived paths
MODEL_DIR = os.path.join(RUN_DIR, "model")
CONFIG_PATH = os.path.join(RUN_DIR, ".hydra/config.yaml")

print(f"Run directory: {RUN_DIR}")
print(f"Config path: {CONFIG_PATH}")

# %% [markdown]
# ## 2. Load Model & Environment

# %%
def setup_model_and_env(config_path, model_dir, model_name="model_final.pkl"):
    """Load trained model and create matching environment."""
    
    # Load config
    config = load_yaml_config(config_path)
    config["encoder"] = config["network"]["encoder"]
    
    # Create environment
    term_keys = config.get("termination_keys", ["offroad", "overlap"])
    env = make_env_for_evaluation(
        max_num_objects=config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=not config.get("waymo_dataset", False),
        observation_type=config["observation_type"],
        observation_config=config["observation_config"],
        termination_keys=term_keys,
    )
    
    # Build network
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    
    networks = ppo_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=3e-4,  # Dummy, not used for inference
        network_config=config
    )
    
    # Load parameters
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
    if "model_final.pkl" in [os.path.basename(f) for f in model_files]:
        model_path = os.path.join(model_dir, "model_final.pkl")
    else:
        model_path = model_files[-1] if model_files else None
    
    if not model_path:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    print(f"Loading model: {model_path}")
    params = load_params(model_path)
    
    return env, networks, params, config

# Load everything
if os.path.exists(CONFIG_PATH):
    env, networks, params, config = setup_model_and_env(CONFIG_PATH, MODEL_DIR)
    print("✓ Model and environment loaded")
else:
    print(f"⚠ Config not found at {CONFIG_PATH}")
    print("  Please update RUN_DIR to point to your training run.")

# %% [markdown]
# ## 3. Attention Extraction Function

# %%
def extract_attention_from_scenario(env, networks, params, scenario, rng_key):
    """
    Run a single scenario through the model and extract attention weights.
    
    Returns:
        dict with 'observations', 'attention_weights', 'actions'
    """
    # Reset environment
    rng_key, reset_key = jax.random.split(rng_key)
    reset_key = jax.random.split(reset_key, 1)  # Batch dim
    env_state = env.reset(scenario, reset_key)
    
    # Get observation
    obs = env_state.observation
    
    # Get the policy network's encoder
    policy_network = networks.policy_network
    
    # We need to call the encoder with return_attention_weights=True
    # The encoder is nested inside the policy network
    # Structure: policy_network.encoder is WayformerEncoder
    
    # Extract encoder params (structure depends on Flax module hierarchy)
    # Usually: params.policy['params']['encoder'] or similar
    
    def forward_with_attention(params, obs):
        """Forward pass that returns attention weights."""
        # Unflatten observation to dict format expected by encoder
        obs_dict = networks.policy_network.unflatten_fn(obs)
        
        # Call encoder directly with attention flag
        encoder = networks.policy_network.encoder
        
        # Get encoder params from full params
        # The exact path depends on your network structure
        # Common patterns:
        #   params.policy['params']['encoder']
        #   params['policy']['encoder']
        
        latent, attention_weights = encoder.apply(
            {'params': params['encoder']},
            obs_dict,
            return_attention_weights=True
        )
        return latent, attention_weights
    
    # JIT compile
    forward_fn = jax.jit(forward_with_attention)
    
    try:
        # Get policy params
        policy_params = params.policy['params']
        latent, attn_weights = forward_fn(policy_params, obs)
        
        return {
            'observation': jax.device_get(obs),
            'latent': jax.device_get(latent),
            'attention_weights': jax.tree_map(lambda x: np.array(jax.device_get(x)), attn_weights),
            'success': True
        }
    except Exception as e:
        print(f"  Error during forward pass: {e}")
        # Try alternative param structure
        try:
            policy_params = params.policy
            latent, attn_weights = forward_fn(policy_params, obs)
            return {
                'observation': jax.device_get(obs),
                'latent': jax.device_get(latent),
                'attention_weights': jax.tree_map(lambda x: np.array(jax.device_get(x)), attn_weights),
                'success': True
            }
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
            return {'success': False, 'error': str(e2)}

# %% [markdown]
# ## 4. Process Scenarios

# %%
def run_extraction(env, networks, params, config, dataset_path, num_scenarios, output_dir):
    """Process multiple scenarios and save attention weights."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generator
    data_gen = make_data_generator(
        path=datasets.get_dataset(dataset_path),
        max_num_objects=config["max_num_objects"],
        include_sdc_paths=not config.get("waymo_dataset", False),
        batch_dims=(1,),  # Single scenario at a time
        seed=42,
        repeat=1,
    )
    
    rng_key = jax.random.PRNGKey(0)
    results = []
    
    print(f"Processing {num_scenarios} scenarios...")
    
    for i, scenario in enumerate(data_gen):
        if i >= num_scenarios:
            break
            
        print(f"  Scenario {i+1}/{num_scenarios}", end="")
        
        # Squeeze batch dim
        scenario = jax.tree_map(lambda x: x.squeeze(0), scenario)
        
        rng_key, extract_key = jax.random.split(rng_key)
        result = extract_attention_from_scenario(env, networks, params, scenario, extract_key)
        
        if result['success']:
            # Save individual result
            save_path = os.path.join(output_dir, f"attention_scenario_{i:04d}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)
            print(f" ✓ Saved to {save_path}")
            results.append(result)
        else:
            print(f" ✗ Failed")
    
    print(f"\n✓ Extracted attention from {len(results)}/{num_scenarios} scenarios")
    return results

# Run extraction
if 'env' in dir():
    results = run_extraction(env, networks, params, config, DATASET_PATH, NUM_SCENARIOS, OUTPUT_DIR)

# %% [markdown]
# ## 5. Analyze Attention Weights

# %%
def analyze_attention(results):
    """Basic analysis of extracted attention weights."""
    
    if not results:
        print("No results to analyze")
        return
    
    # Get attention keys from first result
    sample = results[0]['attention_weights']
    print("Attention weight keys:")
    for key in sample.keys():
        shape = sample[key].shape
        print(f"  {key}: {shape}")
    
    print("\n" + "="*50)
    print("Attention Statistics (averaged over scenarios)")
    print("="*50)
    
    for key in sample.keys():
        weights = [r['attention_weights'][key] for r in results]
        stacked = np.stack(weights)
        
        print(f"\n{key}:")
        print(f"  Shape: {stacked.shape}")
        print(f"  Mean:  {stacked.mean():.4f}")
        print(f"  Std:   {stacked.std():.4f}")
        print(f"  Max:   {stacked.max():.4f}")

if 'results' in dir() and results:
    analyze_attention(results)

# %% [markdown]
# ## 6. Visualize Attention (Optional)

# %%
import matplotlib.pyplot as plt

def plot_attention_heatmap(attention_weights, key, head_idx=0, save_path=None):
    """Plot attention heatmap for a specific layer and head."""
    
    if key not in attention_weights:
        print(f"Key '{key}' not found. Available: {list(attention_weights.keys())}")
        return
    
    attn = attention_weights[key]
    
    # Shape is typically (batch, heads, query, key) or (heads, query, key)
    if attn.ndim == 4:
        attn = attn[0]  # Take first batch
    
    if attn.ndim == 3:
        attn = attn[head_idx]  # Take specific head
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(f'Attention: {key} (Head {head_idx})')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

# Example: Plot first result's attention
if 'results' in dir() and results:
    sample_attn = results[0]['attention_weights']
    first_key = list(sample_attn.keys())[0]
    plot_attention_heatmap(sample_attn, first_key)
