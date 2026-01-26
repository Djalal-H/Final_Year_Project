# Copyright 2025 Valeo.

"""XAI-enabled Proximal Policy Optimization (PPO) trainer with attention extraction.

This trainer extends the standard PPO trainer to periodically extract and log
attention weights for offline Head Specialization Analysis.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.reinforcement import ppo
from vmax.agents.pipeline import inference, pmap
from vmax.agents.networks import encoders, network_utils
from vmax.scripts.training import train_utils
from vmax.simulator import metrics as _metrics


from xai.attention_logger import AttentionLogger


def extract_attention_online(
    env,
    params,
    network_config: dict,
    scenarios,
    n_samples: int = 4,
) -> dict:
    """Extract attention weights from encoder during training.
    
    Processes scenarios sequentially (one at a time) since env.observe()
    doesn't support batched scenarios. This mirrors the approach in
    offline_attention_extraction.ipynb.
    
    Args:
        env: The environment (used for observe and unflatten_fn).
        params: Current training parameters (PPONetworkParams).
        network_config: Network configuration dictionary.
        scenarios: Batched scenarios from training (will process first n_samples).
        n_samples: Number of samples to extract attention for.
        
    Returns:
        Dictionary of attention weights from WayformerEncoder.
        Each value is stacked across all samples.
    """
    # Get unflatten function from environment
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    
    # Parse and convert encoder config
    encoder_cfg = network_config.get('encoder', {})
    encoder_cfg = network_utils.parse_config(encoder_cfg, "encoder")
    encoder_cfg = network_utils.convert_to_dict_with_activation_fn(encoder_cfg)
    
    # Build encoder with return_attention_weights=True
    encoder = encoders.WayformerEncoder(
        unflatten_fn,
        return_attention_weights=True,
        **encoder_cfg
    )
    
    # Extract encoder params from policy params
    if 'params' in params.policy and 'encoder_layer' in params.policy['params']:
        encoder_params = params.policy['params']['encoder_layer']
    else:
        raise ValueError("Could not find 'encoder_layer' in policy params")
    
    # Process scenarios one at a time (env.observe doesn't support batching)
    all_attention_weights = []
    
    print(f"[XAI DEBUG] Processing {n_samples} scenarios for attention extraction...")
    
    # Debug: Check input scenario shapes and structure
    print(f"[XAI DEBUG] Scenario type: {type(scenarios)}")
    if hasattr(scenarios, 'roadgraph_points'):
        print(f"[XAI DEBUG] Input scenarios roadgraph.x shape: {scenarios.roadgraph_points.x.shape}")
    if hasattr(scenarios, 'timestep'):
        print(f"[XAI DEBUG] Input scenarios timestep shape: {scenarios.timestep.shape}")
    if hasattr(scenarios, 'sim_trajectory'):
        print(f"[XAI DEBUG] Input scenarios sim_trajectory.x shape: {scenarios.sim_trajectory.x.shape}")
        print(f"[XAI DEBUG] Input scenarios sim_trajectory.yaw shape: {scenarios.sim_trajectory.yaw.shape}")
    
    for i in range(n_samples):
        print(f"\n[XAI DEBUG] ===== Processing scenario {i+1}/{n_samples} =====")
        
        # Extract single scenario from batch
        # The offline notebook uses squeeze(0) on a batch-1 scenario
        # We need to select index i and ensure all dimensions are properly reduced
        single_scenario = jax.tree_util.tree_map(
            lambda x: x[i] if (hasattr(x, 'ndim') and x.ndim > 0) else x,
            scenarios
        )
        
        # Debug: Check shape after indexing
        print(f"[XAI DEBUG] After indexing [i]:")
        if hasattr(single_scenario, 'roadgraph_points'):
            print(f"[XAI DEBUG]   roadgraph.x shape: {single_scenario.roadgraph_points.x.shape}")
        if hasattr(single_scenario, 'timestep'):
            ts = single_scenario.timestep
            ts_shape = ts.shape if hasattr(ts, 'shape') else 'scalar'
            ts_value = int(ts) if hasattr(ts, 'item') else ts
            print(f"[XAI DEBUG]   timestep shape: {ts_shape}, value: {ts_value}")
        if hasattr(single_scenario, 'sim_trajectory'):
            print(f"[XAI DEBUG]   sim_trajectory.x shape: {single_scenario.sim_trajectory.x.shape}")
            print(f"[XAI DEBUG]   sim_trajectory.yaw shape: {single_scenario.sim_trajectory.yaw.shape}")
        if hasattr(single_scenario, 'object_metadata'):
            if hasattr(single_scenario.object_metadata, 'is_sdc'):
                print(f"[XAI DEBUG]   is_sdc shape: {single_scenario.object_metadata.is_sdc.shape}")
        
        # Now apply squeeze to remove any remaining singleton dimensions
        # BUT preserve timestep as integer if it's a scalar
        def smart_squeeze(x):
            if not hasattr(x, 'squeeze'):
                return x
            # Don't squeeze scalars (0-d arrays)
            if hasattr(x, 'ndim') and x.ndim == 0:
                return x
            return x.squeeze()
        
        single_scenario = jax.tree_util.tree_map(smart_squeeze, single_scenario)
        
        # Debug: Check final shape after squeeze
        print(f"[XAI DEBUG] After squeeze:")
        if hasattr(single_scenario, 'roadgraph_points'):
            print(f"[XAI DEBUG]   roadgraph.x shape: {single_scenario.roadgraph_points.x.shape}")
        if hasattr(single_scenario, 'sim_trajectory'):
            print(f"[XAI DEBUG]   sim_trajectory.yaw shape: {single_scenario.sim_trajectory.yaw.shape}")
        if hasattr(single_scenario, 'timestep'):
            ts = single_scenario.timestep
            ts_shape = ts.shape if hasattr(ts, 'shape') else 'scalar'
            ts_value = int(ts) if hasattr(ts, 'item') else ts
            print(f"[XAI DEBUG]   timestep shape: {ts_shape}, value: {ts_value}")
        
        # Get observation for single scenario
        try:
            obs = env.observe(single_scenario)
            print(f"[XAI DEBUG] ✓ Observation shape: {obs.shape}")
        except Exception as e:
            print(f"[XAI DEBUG] ✗ Error in env.observe for scenario {i}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Add batch dimension for encoder
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
        
        # Run forward pass with attention extraction
        _, attention_weights = encoder.apply({'params': encoder_params}, obs)
        all_attention_weights.append(attention_weights)
        print(f"[XAI DEBUG] ✓ Extracted attention with {len(attention_weights)} keys")
    
    # Stack attention weights across samples
    # Each key will have shape (n_samples, ...) after stacking
    stacked_weights = {}
    if all_attention_weights:
        keys = all_attention_weights[0].keys()
        for key in keys:
            stacked_weights[key] = jnp.concatenate(
                [aw[key] for aw in all_attention_weights],
                axis=0
            )
    
    return stacked_weights



if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env


def train(
    env: waymax_env.PlanningAgentEnvironment,
    data_generator: typing.Iterator[waymax_datatypes.SimulatorState],
    eval_scenario: waymax_datatypes.SimulatorState,
    num_scenario_per_eval: int,
    total_timesteps: int,
    num_envs: int,
    num_episode_per_epoch: int,
    scenario_length: int,
    log_freq: int,
    seed: int,
    value_coef: float,
    entropy_coef: float,
    discount: float,
    gae_lambda: float,
    eps_clip: float,
    normalize_advantages: bool,
    save_freq: int,
    eval_freq: int,
    learning_rate: float,
    grad_updates_per_step: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    network_config: dict,
    progress_fn: Callable[[int, datatypes.Metrics], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    disable_tqdm: bool = False,
    # XAI-specific parameters
    attention_log_freq: int = 1000,
    attention_log_dir: str = "",
    attention_n_samples: int = 4,
) -> None:
    """Train a PPO agent with attention weight extraction for XAI.

    This is an extended version of the standard PPO trainer that periodically
    extracts and logs attention weights for offline Head Specialization Analysis.

    Args:
        env: An instance of the planning environment.
        data_generator: Iterator yielding simulator state samples.
        eval_scenario: Simulator state used for evaluation.
        num_scenario_per_eval: Number of evaluation scenarios.
        total_timesteps: Total training timesteps.
        num_envs: Number of parallel environments.
        num_episode_per_epoch: Episodes per epoch.
        scenario_length: Number of steps per scenario.
        log_freq: Frequency of logging.
        seed: Random seed.
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy loss.
        discount: Discount factor.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        eps_clip: PPO clipping parameter.
        normalize_advantages: Flag for normalizing advantages.
        save_freq: Frequency to save model checkpoints.
        eval_freq: Evaluation frequency.
        learning_rate: Learning rate for optimizers.
        grad_updates_per_step: Gradient update iterations per step.
        batch_size: Batch size.
        unroll_length: Unroll length for trajectory generation.
        num_minibatches: Number of minibatches.
        network_config: Dictionary for network configurations.
        progress_fn: Callback function for reporting progress.
        checkpoint_logdir: Directory path for saving checkpoints.
        disable_tqdm: Flag to disable tqdm progress bar.
        attention_log_freq: Frequency to log attention weights (every N steps).
        attention_log_dir: Directory to save attention logs. If empty, no logging.
        attention_n_samples: Number of samples per batch to log (to save space).

    """
    print(" XAI PPO ".center(40, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1
    do_attention_logging = attention_log_dir != ""

    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    total_iters = (total_timesteps // env_step_per_training_step) + 1

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    network, training_state, policy_fn = ppo.initialize(
        action_size,
        observation_size,
        env,
        learning_rate,
        network_config,
        num_devices,
        network_key,
    )
    learning_fn = ppo.make_sgd_step(
        network,
        num_minibatches,
        gae_lambda,
        discount,
        eps_clip,
        value_coef,
        entropy_coef,
        normalize_advantages,
    )
    step_fn = partial(inference.policy_step, extra_fields=("truncation", "steps", "rewards"))
    print("-> Initializing networks... Done.")
    

    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )

    run_training = partial(
        pipeline.run_training_on_policy,
        env=env,
        learning_fn=learning_fn,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        scan_length=batch_size * num_minibatches // num_envs,
        grad_updates_per_step=grad_updates_per_step,
    )
    run_evaluation = partial(
        pipeline.run_evaluation,
        env=env,
        policy_fn=policy_fn,
        step_fn=step_fn,
        scan_length=scenario_length * num_scenario_per_eval,
    )

    run_training = jax.pmap(run_training, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")

    # Initialize attention logger if enabled
    attention_logger = None
    if do_attention_logging:
        attention_log_path = os.path.join(attention_log_dir, "attention_logs")
        encoder_config = network_config.get('encoder', {})
        attention_logger = AttentionLogger(
            output_dir=attention_log_path,
            config=encoder_config,
        )
        print(f"-> Attention logging enabled. Logging every {attention_log_freq} steps.")
        print(f"   Output: {attention_log_path}")
        print(f"   Samples per log: {attention_n_samples}")

    time_training = perf_counter()

    current_step = 0

    print("-> Ground Control to Major Tom...")
    for iter in tqdm(range(total_iters), desc="XAI Training", total=total_iters, dynamic_ncols=True, disable=disable_tqdm):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        # Batch data generation
        t = perf_counter()
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        # Training step
        t = perf_counter()
        training_state, training_metrics = run_training(batch_scenarios, training_state, iter_keys)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

        epoch_training_time = perf_counter() - t

        #  Log training metrics
        t = perf_counter()
        training_metrics = pmap.flatten_tree(training_metrics)
        training_metrics = jax.device_get(training_metrics)
        training_metrics = _metrics.collect(training_metrics, "ep_len_mean")

        current_step = int(pmap.unpmap(training_state.env_steps))

        metrics = {
            "runtime/sps": int(env_step_per_training_step / epoch_training_time),
            **{f"{name}": value for name, value in training_metrics.items()},
        }

        if do_save and not iter % save_freq:
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            train_utils.save_params(path, pmap.unpmap(training_state.params))

        epoch_log_time = perf_counter() - t

        # Attention logging (XAI) - Online attention extraction
        # Log every N iterations (more reliable than step-based since steps don't align)
        t = perf_counter()
        log_interval_iters = max(1, attention_log_freq // env_step_per_training_step)
        should_log = attention_logger and iter > 0 and iter % log_interval_iters == 0
        if should_log:
            print(f"[XAI] Extracting & logging attention at step {current_step} (iteration {iter})...")
            try:
                # Get sample scenarios for attention extraction
                # batch_scenarios has shape (num_devices, batch_size, ...) or similar
                # We need to flatten to get individual scenarios, then select first n_samples
                
                # First, take scenarios from first device
                device_scenarios = jax.tree_util.tree_map(
                    lambda x: x[0] if x.ndim > 0 else x,
                    batch_scenarios
                )
                
                # Now flatten any remaining batch dimensions to get individual scenarios
                # device_scenarios likely has shape (batch_size, num_minibatches, ...)
                # We need to reshape to (batch_size * num_minibatches, ...)
                def flatten_batch(x):
                    if not hasattr(x, 'ndim') or x.ndim == 0:
                        return x
                    # Flatten first two dimensions if they exist
                    if x.ndim >= 2:
                        new_shape = (-1,) + x.shape[2:]
                        return x.reshape(new_shape)
                    return x
                
                flattened_scenarios = jax.tree_util.tree_map(flatten_batch, device_scenarios)
                
                # Now select first n_samples
                sample_scenarios = jax.tree_util.tree_map(
                    lambda x: x[:attention_n_samples] if (hasattr(x, 'ndim') and x.ndim > 0) else x,
                    flattened_scenarios
                )
                
                # Extract real attention weights online (processes scenarios sequentially)
                current_params = pmap.unpmap(training_state.params)
                attention_weights = extract_attention_online(
                    env, 
                    current_params, 
                    network_config, 
                    sample_scenarios,
                    n_samples=attention_n_samples
                )
                # Convert to numpy for logging
                attention_weights = jax.tree_util.tree_map(
                    lambda x: np.array(jax.device_get(x)), 
                    attention_weights
                )
                
                # Log attention weights with semantic features
                attention_logger.log(
                    step=current_step,
                    attention_weights=attention_weights,
                    simulator_state=sample_scenarios,
                    n_samples=attention_n_samples
                )
                print(f"[XAI] Successfully logged attention for step {current_step}")
                print(f"[XAI] Attention keys: {list(attention_weights.keys())}")
                metrics["xai/attention_log_count"] = current_step // attention_log_freq
                
            except Exception as e:
                print(f"[XAI] Warning: Attention logging failed at step {current_step}: {e}")
                import traceback
                traceback.print_exc()

        epoch_attention_time = perf_counter() - t

        # Evaluation
        t = perf_counter()
        if do_evaluation and not iter % eval_freq:
            eval_metrics = run_evaluation(eval_scenario, training_state)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_metrics)
            eval_metrics = pmap.flatten_tree(eval_metrics)
            eval_metrics = _metrics.collect(eval_metrics, "ep_len_mean")
            progress_fn(current_step, eval_metrics)

        epoch_eval_time = perf_counter() - t

        if not iter % log_freq:
            metrics["runtime/data_time"] = epoch_data_time
            metrics["runtime/training_time"] = epoch_training_time
            metrics["runtime/log_time"] = epoch_log_time
            metrics["runtime/attention_time"] = epoch_attention_time
            metrics["runtime/eval_time"] = epoch_eval_time
            metrics["runtime/iter_time"] = epoch_data_time + epoch_training_time + epoch_log_time + epoch_eval_time + epoch_attention_time
            metrics["runtime/wall_time"] = perf_counter() - time_training
            metrics["train/rl_gradient_steps"] = int(pmap.unpmap(training_state.rl_gradient_steps))
            metrics["train/env_steps"] = current_step

            progress_fn(current_step, metrics, total_timesteps)

            if disable_tqdm:
                print(f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%")

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= total_timesteps

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        train_utils.save_params(path, pmap.unpmap(training_state.params))

    # Close attention logger
    if attention_logger:
        attention_logger.close()
        print("[XAI] Attention logging complete.")

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
