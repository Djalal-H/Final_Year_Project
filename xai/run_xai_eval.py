
import argparse
import json
import math
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from vmax.scripts.evaluate import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator

# Counterfactual Explainer for Causal XRL
from xai.counterfactual_explainer import (
    CounterfactualExplainer,
    CounterfactualConfig,
    format_counterfactual_rationale,
)

# Disable GPU preallocation for JAX if needed (usually good practice in mixed envs)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# =============================================================================
# SEMANTIC GRAPH BUILDER (Adapted from xai/build_semantic_graph_v2.py)
# =============================================================================

class SemanticGraphBuilder:
    def __init__(self, lane_width_m: float = 3.5): # Standard lane width is usually ~3.5m
        self.lane_width_m = lane_width_m

    def ego_frame(self, dx: float, dy: float, ego_yaw: float) -> Tuple[float, float]:
        """Convert world delta (dx,dy) into ego-local coordinates: x_forward, y_left."""
        c = math.cos(-ego_yaw)
        s = math.sin(-ego_yaw)
        x_local = dx * c - dy * s
        y_local = dx * s + dy * c
        return x_local, y_local

    def distance_bucket(self, dist: float) -> str:
        if dist < 10.0: return "very_close"
        if dist < 30.0: return "close"
        if dist < 50.0: return "medium_range"
        return "far"

    def ttc_bucket(self, ttc: float) -> str:
        if ttc < 2.0: return "collision_imminent"
        if ttc < 5.0: return "collision_likely"
        return "safe_for_now"

    def compute_closing_speed(self, ex, ey, evx, evy, ox, oy, ovx, ovy) -> Tuple[float, float]:
        rel = np.array([ox - ex, oy - ey], dtype=np.float32)
        dist = float(np.linalg.norm(rel) + 1e-6)
        
        ego_v = np.array([evx, evy], dtype=np.float32)
        oth_v = np.array([ovx, ovy], dtype=np.float32)
        rel_v = oth_v - ego_v
        
        # closing speed: positive means getting closer
        closing = float(-np.dot(rel, rel_v) / dist)
        return closing, dist

    def determine_action_state(self, vx, vy) -> str:
        # Simple heuristic since we lack acceleration data in this specific slice
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 0.1: return "stopped"
        return "moving"

    def build(self, state_dict: Dict[str, Any], ego_idx: int) -> Dict[str, Any]:
        cur = state_dict["current"]
        N = len(cur["x"])
        
        # 1. Extract Ego Data
        ex, ey = float(cur["x"][ego_idx]), float(cur["y"][ego_idx])
        evx, evy = float(cur["vx"][ego_idx]), float(cur["vy"][ego_idx])  # <--- FIXED
        ego_yaw = float(cur["yaw"][ego_idx])
        ego_speed = math.sqrt(evx**2 + evy**2)

        # 2. Prepare Containers matching your Desired Output
        scenario_info = {
            "ego_velocity": round(ego_speed, 2),
            "ego_action": self.determine_action_state(evx, evy)
        }
        context_nodes = []
        semantic_edges = []
        
        valid_indices = [i for i in range(N) if cur["valid"][i]]

        # 3. Build Edges & Nodes
        for j in valid_indices:
            if j == ego_idx: continue

            # Extract Other Agent Data
            ox, oy = float(cur["x"][j]), float(cur["y"][j])
            ovx, ovy = float(cur["vx"][j]), float(cur["vy"][j])
            otype = "vehicle" # Default, or map from state_dict['type'] if available
            
            # Calculations
            closing, dist = self.compute_closing_speed(ex, ey, evx, evy, ox, oy, ovx, ovy)
            dx, dy = ox - ex, oy - ey
            x_local, y_local = self.ego_frame(dx, dy, ego_yaw)
            
            # Semantic Tagging
            relations = []
            relations.append(self.distance_bucket(dist))
            
            # Spatial Relations
            if x_local > 2.0: relations.append("ahead")
            elif x_local < -2.0: relations.append("behind")
            else: relations.append("beside")

            if y_local > 1.0: relations.append("to_the_left")
            elif y_local < -1.0: relations.append("to_the_right")
            
            # Dynamic Relations
            if closing > 1.0: # Only flag approaching if significant closing speed
                relations.append("approaching")
                ttc = dist / closing
                relations.append(self.ttc_bucket(ttc))
            elif closing < -1.0:
                relations.append("moving_away")
            
            # Lane Context
            if abs(y_local) < self.lane_width_m:
                 relations.append("in_ego_lane")

            # Add to Edges
            semantic_edges.append({
                "target_id": int(j),
                "target_type": otype,
                "relation": relations,
                "raw_distance": dist
            })

            # Add to Context Nodes (include dist for easy lookup)
            context_nodes.append({
                "id": int(j),
                "type": otype,
                "distance_to_ego": round(dist, 1),
                "speed": round(math.sqrt(ovx**2 + ovy**2), 1)
            })

        # Return the exact structure your prompt generator needs
        return {
            "scenario_info": scenario_info,
            "context_nodes": context_nodes,
            "semantic_edges": semantic_edges
        }
# =============================================================================
# LLM PROMPT GENERATOR
# =============================================================================

def generate_explanation_prompt(
    graph_json: Dict[str, Any], 
    counterfactual_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Converts the semantic graph JSON into a structured Natural Language Prompt.
    Optionally includes counterfactual decision rationale if provided.
    
    Args:
        graph_json: Semantic graph with scenario_info, context_nodes, semantic_edges
        counterfactual_result: Optional output from CounterfactualExplainer.explain()
        
    Returns:
        Formatted prompt string for LLM
    """
    ego = graph_json['scenario_info']
    edges = graph_json['semantic_edges']
    nodes = graph_json['context_nodes']

    # 1. Set the Scene
    prompt = (
        f"You are the explanatory module for an Autonomous Driving RL Agent.\n"
        f"Current State:\n"
        f"- The Ego Vehicle is {ego['ego_action']} at {ego['ego_velocity']} m/s.\n"
        f"Describe the critical factors in the scene based on the following observations:\n"
    )

    # 2. Filter for Critical Edges
    # Priority: "in_ego_lane", "approaching", "very_close", "collision_likely"
    CRITICAL_TAGS = {"in_ego_lane", "approaching", "very_close", "collision_likely", "collision_imminent"}
    
    critical_descriptions = []

    # Sort edges by distance so the list is ordered by proximity
    edges.sort(key=lambda x: x['raw_distance'])

    for edge in edges:
        rels = set(edge['relation'])
        
        # Check if this edge has any critical tag
        if not rels.isdisjoint(CRITICAL_TAGS):
            
            # Fetch node details (speed, etc.)
            agent_id = edge['target_id']
            agent_type = edge['target_type']
            node_data = next((n for n in nodes if n['id'] == agent_id), None)
            
            # Format the Relation String (clean up underscores)
            readable_rels = ", ".join([r.replace("_", " ") for r in edge['relation']])
            
            desc = (f"- {agent_type.capitalize()} {agent_id} is {node_data['distance_to_ego']}m away. "
                    f"Status: [{readable_rels}].")
            
            critical_descriptions.append(desc)

    # 3. Add Observations to Prompt
    if critical_descriptions:
        prompt += "\n".join(critical_descriptions)
    else:
        prompt += "- No critical traffic nearby."

    # 4. Add Decision Rationale (Causal XRL) if counterfactual data is available
    if counterfactual_result is not None:
        rationale = format_counterfactual_rationale(counterfactual_result)
        if rationale:
            prompt += rationale

    # 5. Ask for the "Why" (The Chain of Thought trigger)
    prompt += (
        "\n\nBased on these observations, explain why the agent might choose "
        "to maintain speed, decelerate, or change lanes."
    )
    
    return prompt

# =============================================================================
# GRAPH VISUALIZATION (adapted from viz_record_graph.py)
# =============================================================================
import matplotlib.pyplot as plt
from pathlib import Path

def plot_graph_on_scene(
    state, 
    graph: Dict[str, Any], 
    ego_idx: int, 
    output_path: str,
    max_dist: float = 80.0,
    show_ids: bool = True
):
    """
    Plots the current scene with agents, roadgraph (if available), and semantic graph edges.
    Visualization style adapted from viz_record_graph_v1.py.
    Saves the plot to `output_path`.
    
    Args:
        state: The SimulatorState from v-max/waymax.
        graph: The semantic graph dict with 'context_nodes' and 'semantic_edges'.
        ego_idx: Index of the ego vehicle in the state arrays.
        output_path: Path to save the PNG.
        max_dist: Only draw edges for agents within this distance.
        show_ids: If True, show agent IDs as text labels.
    """
    # Get timestep and trajectory data
    def get_aa(x):
        arr = np.array(jax.device_get(x))
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    idx = int(get_aa(state.timestep))
    traj = state.sim_trajectory
    
    def get_field(tk):
        val = jax.device_get(tk)
        if val.ndim == 3:
            return np.array(val[0, :, idx])
        elif val.ndim == 2:
            return np.array(val[:, idx])
        return np.array(val)

    cur_x = get_field(traj.x)
    cur_y = get_field(traj.y)
    cur_yaw = get_field(traj.yaw)
    cur_valid = get_field(traj.valid).astype(bool)

    # Roadgraph points (if available)
    rg_xyz = None
    rg_valid = None
    if hasattr(state, 'roadgraph_points'):
        rg = state.roadgraph_points
        if hasattr(rg, 'xyz') and hasattr(rg, 'valid'):
            rg_xyz_raw = jax.device_get(rg.xyz)
            rg_valid_raw = jax.device_get(rg.valid)
            # Shape usually (Batch, P, 3) or (P, 3)
            if rg_xyz_raw.ndim == 3:
                rg_xyz = np.array(rg_xyz_raw[0])
                rg_valid = np.array(rg_valid_raw[0]).astype(bool)
            else:
                rg_xyz = np.array(rg_xyz_raw)
                rg_valid = np.array(rg_valid_raw).astype(bool)

    # Create figure (matching viz_record_graph_v1.py style)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # Plot roadgraph points (matching viz_record_graph_v1.py: s=1, alpha=0.3)
    if rg_xyz is not None and rg_valid is not None:
        pts = rg_xyz[rg_valid]
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3)

    # Plot agents (matching viz_record_graph_v1.py: s=25)
    valid_idx = np.where(cur_valid)[0]
    ax.scatter(cur_x[valid_idx], cur_y[valid_idx], s=25)

    # Highlight ego (matching viz_record_graph_v1.py: s=120, marker="*")
    ex, ey = float(cur_x[ego_idx]), float(cur_y[ego_idx])
    ego_yaw = float(cur_yaw[ego_idx])
    ax.scatter([ex], [ey], s=120, marker="*")

    # Show agent IDs as text labels (matching viz_record_graph_v1.py)
    if show_ids:
        for i in valid_idx:
            ax.text(cur_x[i], cur_y[i], str(int(i)), fontsize=8)
    
    # Heading arrow for ego (matching viz_record_graph_v1.py)
    ax.arrow(
        ex, ey,
        5.0 * math.cos(ego_yaw), 5.0 * math.sin(ego_yaw),
        head_width=1.5, length_includes_head=True
    )

    # Overlay graph edges + sanity checks (matching viz_record_graph_v1.py)
    mismatches = []
    edges = graph.get("semantic_edges", [])
    
    for e_ in edges:
        dst = int(e_["target_id"])
        dist = float(e_.get("raw_distance", 0.0))
        if dist > max_dist:
            continue
        
        # Draw edge (matching viz_record_graph_v1.py: linewidth=1, alpha=0.6)
        ax.plot([ex, float(cur_x[dst])], [ey, float(cur_y[dst])], linewidth=1, alpha=0.6)
        
        # Check label consistency (adapted from viz_record_graph_v1.py)
        rels = set(e_.get("relation", []))
        
        # Compute local coordinates for sanity check
        dx, dy = float(cur_x[dst]) - ex, float(cur_y[dst]) - ey
        c = math.cos(-ego_yaw)
        s = math.sin(-ego_yaw)
        x_local = dx * c - dy * s
        y_local = dx * s + dy * c
        
        # Get closing speed from graph if available (we store it in relation tags)
        closing = 0.0  # Default, since we don't store raw closing speed in edges
        
        # Mismatch checks (adapted from viz_record_graph_v1.py)
        if ("ahead" in rels) and not (x_local > 0):
            mismatches.append((dst, "ahead", x_local, y_local, closing))
        if ("behind" in rels) and not (x_local < 0):
            mismatches.append((dst, "behind", x_local, y_local, closing))
        if ("to_the_left" in rels) and not (y_local > 0):
            mismatches.append((dst, "to_the_left", x_local, y_local, closing))
        if ("to_the_right" in rels) and not (y_local < 0):
            mismatches.append((dst, "to_the_right", x_local, y_local, closing))
        if ("approaching" in rels):
            # We have 'approaching' tag but would need raw closing speed to fully validate
            pass
        if ("moving_away" in rels):
            pass

    # Title format matching viz_record_graph_v1.py
    ax.set_title(f"nodes={len(graph.get('context_nodes', []))}, edges={len(edges)}, ego={ego_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    
    # Print mismatches (matching viz_record_graph_v1.py)
    if mismatches:
        print(f"    Mismatches found: {len(mismatches)}")
        for m in mismatches[:20]:
            dst, lab, xl, yl, cl = m
            print(f"      dst={dst:3d} label={lab:12s} x_local={xl:7.2f} y_local={yl:7.2f} closing={cl:6.2f}")
        if len(mismatches) > 20:
            print("      ... (truncated)")
    
    print(f"    -> Saved graph plot: {out_path}")


# =============================================================================
# EVALUATION & LOGGING LOOP
# =============================================================================


def state_to_dict(state) -> Dict[str, Any]:
    """
    Convert JAX SimulatorState to a Python dictionary of numpy arrays for the graph builder.
    """
    # Assuming state structure matches Waymax/V-Max conventions
    # We transfer to CPU (numpy) first
    
    # NOTE: shape is (Batch, N_objects, T) for trajectory fields usually, or (N_objects, T) if squeezed?
    # We squeezed the tree earlier so batch dim should be gone effectively if we handled it right in the loop,
    # BUT `state_to_dict` receives `env_transition.state`.
    # `env_transition` comes from `jitted_reset` or `jitted_step`.
    # If `BraxWrapper` returns (1, ...) shapes, then we need to be careful.
    # The user said the reshape was needed for reset input, so env output is likely (1, ...).
    
    # We will assume batch dimension 0 exists and we want the 0-th element.
    # OR if we vmapped, it is indeed batched.
    
    def get_aa(x):
        # device_get returns numpy array.
        # If x is scalar-like but shaped (1,), [0] gets the value.
        return np.array(jax.device_get(x))[0]

    # Get timestep index
    # validation: idx might be scalar or (1,)
    idx = int(get_aa(state.timestep))
    
    # Access trajectory
    traj = state.sim_trajectory
    
    # helper for (1, N, T) -> (N,) at current time
    # Or (N, T) -> (N,)
    def get_field(tk):
        # tk is likely (1, N, T) or (N, T)
        val = jax.device_get(tk)
        if val.ndim == 3: # (Batch, N, T)
            return np.array(val[0, :, idx])
        elif val.ndim == 2: # (N, T)
            return np.array(val[:, idx])
        return np.array(val) # fallback?

    return {
        "current": {
            "x": get_field(traj.x),
            "y": get_field(traj.y),
            "vx": get_field(traj.vel_x),
            "vy": get_field(traj.vel_y),
            "yaw": get_field(traj.yaw),
            "valid": get_field(traj.valid),
        }
    }

def run_xai_eval(args):
    print(f"-> Setting up evaluation for {args.sdc_actor}...")
    
    # 1. Setup Env & Policy
    # Reusing V-Max setup utility
    env, step_fn, eval_path, termination_keys = utils.setup_evaluation(
        args.sdc_actor,
        args.path_model,
        args.src_dir,
        args.path_dataset,
        args.eval_name,
        args.max_num_objects,
        args.noisy_init,
        sdc_paths_from_data=(not args.waymo_dataset), # standard default in eval script
    )
    
    # 2. Data Generator
    batch_dims = (1, 1) # Force batch size 1 for this detailed logging
    data_generator = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=(not args.waymo_dataset),
        batch_dims=batch_dims,
        seed=args.seed,
        repeat=1,
    )

    # 3. Helpers
    graph_builder = SemanticGraphBuilder()
    
    # Counterfactual Explainer for Causal XRL
    cf_config = CounterfactualConfig(
        horizon_steps=args.cf_horizon,  # Configurable rollout horizon
    )
    cf_explainer = CounterfactualExplainer(config=cf_config)
    
    results_log = []
    
    # JIT the step function
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)
    
    rng_key = jax.random.PRNGKey(args.seed)
    
    scenarios_processed = 0
    
    print(f"-> Starting loop. Limit: {args.limit} scenarios.")
    
    for scenario_idx, scenario in enumerate(data_generator):
        if scenarios_processed >= args.limit:
            break
            
        # Optional: Specific filtering
        if args.scenario_indexes and scenario_idx not in args.scenario_indexes:
            continue
            
        print(f"Processing Scenario {scenario_idx}...")
        
        # Squeeze batch dim if present (1,) -> ()
        scenario = jax.tree_map(lambda x: x.squeeze(0), scenario)

        # Reset

        rng_key, reset_key = jax.random.split(rng_key)
        # Handle shape mismatch if needed, similar to utils.run_scenario_jit
        reset_key = jax.random.split(reset_key, 1) # batch size 1
        
        env_transition = jitted_reset(scenario, reset_key)
        
        scenario_events = []
        episode_images = []
        
        done = False
        step_count = 0
        
        # SDC index: usually the first valid one or marked, Waymax usually assumes idx same as dataloader
        # But we need to find it from state or assume known. 
        # For V-Max/Waymax, SD is usually index which `is_sdc` is true.
        # We need to extract `is_sdc` from the scenario or state if available.
        # In `state_to_dict` we only grabbed kinematics. 
        # Typically SDC is index provided by metadata, but let's assume we can find it.
        # For simplicity, we will check `is_sdc` from the scenario inputs (Observation/State).
        # Actually `env_transition.state` likely has `is_sdc` info.
        
        # Let's extract is_sdc once per scenario
        # is_sdc is in object_metadata.is_sdc
        # shape usually (Batch, N) or (N,)
        is_sdc_raw = jax.device_get(env_transition.state.object_metadata.is_sdc)
        if is_sdc_raw.ndim == 2: # (Batch, N)
             is_sdc_arr = np.array(is_sdc_raw)[0]
        else:
             is_sdc_arr = np.array(is_sdc_raw)
        ego_idx = int(np.argmax(is_sdc_arr))
        
        while not done:
            # A. Build Graph
            # Convert JAX state to dict
            st_dict = state_to_dict(env_transition.state)
            
            # Add types if we can (placeholder)
            # st_dict["current"]["type"] = ...
            
            graph = graph_builder.build(st_dict, ego_idx)
            
            # B. Counterfactual Explanation (Causal XRL)
            counterfactual_result = None
            if args.enable_counterfactuals and step_count % 20 == 0:
                try:
                    # Use vectorized version for faster performance (parallel rollouts)
                    print(f"    [XAI] Generating counterfactuals (Mode: Vectorized/Parallel)...")
                    counterfactual_result = cf_explainer.explain_vectorized(
                        env_transition=env_transition,
                        env=env,
                        step_fn=jitted_step_fn,
                        chosen_action_idx=0  # Default: maintain speed + lane keep
                    )
                except Exception as e:
                    print(f"    ⚠️ Parallel counterfactual generation failed: {e}. Falling back to sequential.")
                    try:
                        print(f"    [XAI] Generating counterfactuals (Mode: Sequential)...")
                        counterfactual_result = cf_explainer.explain(
                            env_transition=env_transition,
                            env=env,
                            step_fn=jitted_step_fn,
                            chosen_action_idx=0
                        )
                    except Exception as e2:
                        print(f"    ⚠️ Sequential counterfactual generation also failed: {e2}")
                        counterfactual_result = None
            
            # C. Periodic LLM Prompt & Graph Visualization
            llm_prompt = None
            if step_count % 20 == 0: # 0.5Hz at 10Hz sim
                llm_prompt = generate_explanation_prompt(graph, counterfactual_result)
                
                # Save graph visualization
                graph_plot_path = os.path.join(
                    eval_path, 
                    "graph_plots", 
                    f"scenario_{scenario_idx}_step_{step_count}.png"
                )
                plot_graph_on_scene(
                    env_transition.state, 
                    graph, 
                    ego_idx, 
                    graph_plot_path
                )
            
            # D. Log
            scenario_events.append({
                "step": step_count,
                "timestamp": round(step_count * 0.1, 2),
                "graph": graph,
                "counterfactual": counterfactual_result,
                "llm_prompt": llm_prompt
            })
            
            # D. Render (if requested)
            if args.render:
                # Use standard V-Max plotting
                # plot_scene returns an image array (H, W, 3)
                img = utils.plot_scene(env, env_transition, sdc_pov=False)
                episode_images.append(img)
            
            # E. Step
            rng_key, step_key = jax.random.split(rng_key)
            step_key = jax.random.split(step_key, 1)
            env_transition, _ = jitted_step_fn(env_transition, key=step_key)
            
            done = bool(jax.device_get(env_transition.done)[0])
            step_count += 1
            
        # End of Scenario
        result_entry = {
            "scenario_id": f"scenario_{scenario_idx}", # or real ID if available
            "events": scenario_events
        }
        results_log.append(result_entry)
        
        # Save Video
        if args.render and episode_images:
             utils.write_video(eval_path, episode_images, idx=scenario_idx)

        scenarios_processed += 1

    # 4. Save Logs
    out_file = os.path.join(eval_path, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results_log, f, indent=2)
        
    print(f"✅ Evaluation finished. Logs saved to {out_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic Eval Args
    parser.add_argument("--sdc_actor", default="expert", help="Actor type")
    parser.add_argument("--path_model", default="", help="Model identifier")
    parser.add_argument("--path_dataset", default="local_womd_valid")
    parser.add_argument("--limit", type=int, default=20, help="Max scenarios")
    parser.add_argument("--render", type=str2bool, default=False)
    
    # Boilerplate needed for setup_evaluation
    parser.add_argument("--src_dir", default="runs")
    parser.add_argument("--eval_name", default="xai_eval")
    parser.add_argument("--max_num_objects", type=int, default=64)
    parser.add_argument("--noisy_init", type=str2bool, default=False)
    parser.add_argument("--waymo_dataset", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario_indexes", nargs="*", type=int)
    
    # Counterfactual Explainer Args
    parser.add_argument("--enable_counterfactuals", type=str2bool, default=True,
                        help="Enable counterfactual explanations (Causal XRL)")
    parser.add_argument("--cf_horizon", type=int, default=20,
                        help="Counterfactual rollout horizon in steps (default: 20 = 2.0s)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_xai_eval(args)
