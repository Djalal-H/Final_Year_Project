"""Multi-timestep Causal Steering with Rollout Comparison.

For each scenario and steering temperature α, two full episode rollouts are run
from the same initial state:

  • Baseline rollout  — agent acts normally (no SAE intervention).
  • Steered rollout   — at every timestep the target SAE feature is clamped
                        before the policy head computes the action.

Both rollouts are evaluated with the full vmax metrics suite.  The difference
in aggregate scores (Δscore = steered − baseline) is the causal fingerprint of
the feature: a feature that reliably improves safety when activated but was
previously weakly expressed is a candidate for a "beneficial safety concept".

Key implementation note
    The environment state diverges after the first steered action.  That's
    expected and desired — the whole point is to see how a single feature's
    influence compounds through time.  Do NOT reset the environment between
    timesteps.

Usage:
    python -m sae_interpretability.rollout_steering \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --sae_checkpoint sae_model.pt \\
        --feature_idx 42 \\
        --temperatures -5.0 -2.0 -1.0 1.0 2.0 5.0 \\
        --n_scenarios 5 \\
        --max_steps 80 \\
        --output data/sae_interpretability/rollout_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from waymax import datatypes as waymax_datatypes

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sae_interpretability.causal_steering import CausalSteerer
from sae_interpretability.config import SAEConfig
from vmax.simulator.metrics import (
    _VMAX_METRICS_REGISTRY,
    AtFaultCollisionMetric,
    ComfortMetric,
    DrivingDirectionComplianceMetric,
    OnMultipleLanesMetric,
    ProgressRatioMetric,
    RunRedLightMetric,
    SpeedLimitViolationMetric,
    TimeToCollisionMetric,
)


# ---------------------------------------------------------------------------
# Metrics collected per rollout step
# ---------------------------------------------------------------------------

_STEP_METRICS: Dict[str, Any] = {
    "run_red_light":               RunRedLightMetric(),
    "ttc":                         TimeToCollisionMetric(),
    "at_fault_collision":          AtFaultCollisionMetric(),
    "comfort":                     ComfortMetric(),
    "speed_limit":                 SpeedLimitViolationMetric(),
    "on_multiple_lanes":           OnMultipleLanesMetric(),
    "driving_direction_compliance": DrivingDirectionComplianceMetric(),
    "progress_ratio_nuplan":       ProgressRatioMetric(),
}


def _compute_step_metrics(state) -> Dict[str, float]:
    """Compute all registered vmax metrics for a single simulator state.

    Returns:
        Dict mapping metric name → scalar float value for this timestep.
    """
    step_vals: Dict[str, float] = {}
    for name, metric in _STEP_METRICS.items():
        try:
            result = metric.compute(state)
            # MetricResult has .value (array) and .valid (bool array)
            val = np.array(jax.device_get(result.value))
            valid = np.array(jax.device_get(result.valid))
            if valid.any():
                step_vals[name] = float(np.mean(val[valid]))
            else:
                step_vals[name] = float("nan")
        except Exception as exc:
            step_vals[name] = float("nan")
    return step_vals


def _aggregate_episode_metrics(
    per_step: List[Dict[str, float]]
) -> Dict[str, float]:
    """Aggregate per-timestep metric dicts into episode-level scalars.

    Uses the same semantics as the vmax collector:
      - overlap / offroad / run_red_light → max  (worst case)
      - ttc, comfort, speed_limit, lanes  → mean
      - progress                           → final value
    """
    if not per_step:
        return {}

    keys = per_step[0].keys()
    agg: Dict[str, float] = {}

    _max_keys  = {"run_red_light", "at_fault_collision", "overlap", "offroad"}
    _final_keys = {"progress_ratio_nuplan", "sdc_progression"}

    for k in keys:
        vals = [s[k] for s in per_step if not np.isnan(s.get(k, float("nan")))]
        if not vals:
            agg[k] = float("nan")
        elif k in _max_keys:
            agg[k] = float(np.max(vals))
        elif k in _final_keys:
            agg[k] = float(vals[-1])
        else:
            agg[k] = float(np.mean(vals))

    return agg


# ---------------------------------------------------------------------------
# RolloutSteerer
# ---------------------------------------------------------------------------

class RolloutSteerer(CausalSteerer):
    """Run paired baseline / steered full-episode rollouts and compare metrics.

    Inherits all SAE machinery from CausalSteerer.  Adds:
      - _policy_action()   : get action from current obs via policy head.
      - _steered_action()  : same but with SAE feature clamped first.
      - run_paired_rollout(): execute both arms and collect metrics.
      - run_rollout_experiment(): outer loop over scenarios and temperatures.
    """

    # ------------------------------------------------------------------
    # Policy action helpers
    # ------------------------------------------------------------------

    def _get_h(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation → residual stream vector h  [D]."""
        if obs.ndim == 1:
            obs = obs[None]
        h, _ = self.encoder.apply({'params': self.encoder_params}, obs)
        if h.ndim > 1:
            h = h[0]
        return h  # [D]

    def _policy_from_h(self, h: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Run policy FC head on h → raw action vector, or None."""
        return self._apply_fc_head(
            h[None], self._policy_fc_params, self._policy_dense_params
        )

    def _steer_h(self, h: jnp.ndarray, feature_idx: int, alpha: float) -> jnp.ndarray:
        """Apply SAE causal intervention on h and return modified h."""
        f_base = self._sae_encode(h)
        h_reconstructed = self._sae_decode(f_base)

        f_steered = f_base.at[feature_idx].add(alpha)
        h_steered_sae = self._sae_decode(f_steered)

        delta = (h_steered_sae - h_reconstructed) * self._act_std_jnp
        return h + delta  # [D]

    def _action_from_policy_output(
        self, policy_out: jnp.ndarray, env_transition
    ) -> waymax_datatypes.Action:
        """Wrap a raw policy vector into a Waymax Action object.

        The InvertibleBicycleModel expects actions of shape (batch, 2):
        [acceleration, steering].
        """
        action_np = np.array(jax.device_get(policy_out)).ravel()[:2]
        action_data = jnp.array(action_np[None], dtype=jnp.float32)  # [1, 2]
        valid_mask = jnp.ones((1, 1), dtype=jnp.bool_)
        action = waymax_datatypes.Action(data=action_data, valid=valid_mask)
        action.validate()
        return action

    # ------------------------------------------------------------------
    # Single paired rollout
    # ------------------------------------------------------------------

    def run_paired_rollout(
        self,
        scenario_batch,
        feature_idx: int,
        alpha: float,
        max_steps: int = 80,
    ) -> Dict[str, Any]:
        """Run baseline and steered rollouts from the same initial state.

        Both rollouts start from the same env.reset() call. They step
        independently: the baseline uses the unmodified policy action and the
        steered run uses the intervention-modified action at every timestep.

        Key invariant: we DO NOT reset the environment between timesteps in
        either rollout. The environment state diverges naturally as the two
        agents take different actions, which is exactly what we want to measure.

        Args:
            scenario_batch: Batched scenario from the data generator.
            feature_idx: SAE feature index to steer.
            alpha: Temperature (additive intervention magnitude).
            max_steps: Maximum episode length.

        Returns:
            Dict with per-step records and aggregated metrics for both arms.
        """
        # ── Baseline rollout ──────────────────────────────────────────────
        baseline_steps: List[Dict] = []
        baseline_transition = self.env.reset(scenario_batch)

        for t in range(max_steps):
            obs = baseline_transition.observation
            h = self._get_h(obs)
            policy_out = self._policy_from_h(h)

            step_record: Dict[str, Any] = {'t': t}
            if policy_out is not None:
                pa = np.array(jax.device_get(policy_out)).ravel()
                step_record['accel'] = float(pa[0]) if len(pa) > 0 else None
                step_record['steer'] = float(pa[1]) if len(pa) > 1 else None
                step_record['policy'] = pa.tolist()

            # Collect metrics for this state
            state_sq = jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if hasattr(x, 'ndim') and x.ndim > 0 else x,
                baseline_transition.state,
            )
            step_record['metrics'] = _compute_step_metrics(state_sq)
            baseline_steps.append(step_record)

            # Check termination
            if t > 0 and bool(jax.device_get(baseline_transition.done)):
                break

            # Step with baseline action
            if policy_out is not None:
                action = self._action_from_policy_output(policy_out, baseline_transition)
                baseline_transition = self.env.step(baseline_transition, action)
            else:
                break

        # ── Steered rollout ───────────────────────────────────────────────
        steered_steps: List[Dict] = []
        steered_transition = self.env.reset(scenario_batch)

        for t in range(max_steps):
            obs = steered_transition.observation
            h = self._get_h(obs)
            h_mod = self._steer_h(h, feature_idx, alpha)
            policy_out = self._policy_from_h(h_mod)

            step_record_s: Dict[str, Any] = {
                't': t,
                'feature_activation_before': float(self._sae_encode(h)[feature_idx]),
                'feature_activation_after':  float(self._sae_encode(h)[feature_idx] + alpha),
            }
            if policy_out is not None:
                pa = np.array(jax.device_get(policy_out)).ravel()
                step_record_s['accel'] = float(pa[0]) if len(pa) > 0 else None
                step_record_s['steer'] = float(pa[1]) if len(pa) > 1 else None
                step_record_s['policy'] = pa.tolist()

            state_sq = jax.tree_util.tree_map(
                lambda x: x.squeeze(0) if hasattr(x, 'ndim') and x.ndim > 0 else x,
                steered_transition.state,
            )
            step_record_s['metrics'] = _compute_step_metrics(state_sq)
            steered_steps.append(step_record_s)

            if t > 0 and bool(jax.device_get(steered_transition.done)):
                break

            if policy_out is not None:
                action = self._action_from_policy_output(policy_out, steered_transition)
                steered_transition = self.env.step(steered_transition, action)
            else:
                break

        # ── Aggregate ─────────────────────────────────────────────────────
        baseline_metrics = _aggregate_episode_metrics(
            [s['metrics'] for s in baseline_steps]
        )
        steered_metrics = _aggregate_episode_metrics(
            [s['metrics'] for s in steered_steps]
        )

        # Compute delta for every metric
        delta_metrics: Dict[str, float] = {}
        for k in baseline_metrics:
            bv = baseline_metrics[k]
            sv = steered_metrics.get(k, float('nan'))
            if not (np.isnan(bv) or np.isnan(sv)):
                delta_metrics[k] = sv - bv
            else:
                delta_metrics[k] = float('nan')

        return {
            'alpha': alpha,
            'feature_idx': feature_idx,
            'n_steps_baseline': len(baseline_steps),
            'n_steps_steered':  len(steered_steps),
            'baseline_metrics': baseline_metrics,
            'steered_metrics':  steered_metrics,
            'delta_metrics':    delta_metrics,
            'per_step_baseline': baseline_steps,
            'per_step_steered':  steered_steps,
        }

    # ------------------------------------------------------------------
    # Outer experiment loop
    # ------------------------------------------------------------------

    def run_rollout_experiment(
        self,
        feature_idx: int,
        temperatures: List[float],
        n_scenarios: int = 5,
        max_steps: int = 80,
        output_path: str = "data/sae_interpretability/rollout_results.json",
    ) -> Dict[str, Any]:
        """Run paired rollouts across multiple scenarios and temperatures.

        For each (scenario, α) pair, two full episodes are executed.  Results
        are aggregated across scenarios and saved to JSON.  Action trajectory
        plots are saved alongside the JSON.

        Args:
            feature_idx:  SAE feature to steer.
            temperatures: List of α values to test.
            n_scenarios:  Number of independent scenarios to evaluate.
            max_steps:    Maximum episode length per rollout.
            output_path:  Output JSON file path.

        Returns:
            Full results dict.
        """
        from vmax.simulator import make_data_generator, datasets

        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        all_scenarios: List[Dict] = []
        print(
            f"\n[RolloutSteerer] Feature {feature_idx}  "
            f"α={temperatures}  {n_scenarios} scenarios  max_steps={max_steps}"
        )

        for scen_idx, scenario_batch in enumerate(data_gen):
            if scen_idx >= n_scenarios:
                break

            print(f"\n  ── Scenario {scen_idx + 1}/{n_scenarios} ──")
            scenario_results: List[Dict] = []

            for alpha in temperatures:
                print(f"    α={alpha:+.2f} …", end="", flush=True)
                result = self.run_paired_rollout(
                    scenario_batch, feature_idx, alpha, max_steps
                )
                result['scenario_id'] = scen_idx
                scenario_results.append(result)

                nb = result['n_steps_baseline']
                ns = result['n_steps_steered']
                dm = result['delta_metrics']
                print(
                    f" done  baseline={nb}t  steered={ns}t  "
                    + "  ".join(
                        f"Δ{k}={v:+.3f}"
                        for k, v in dm.items()
                        if not np.isnan(v)
                    )
                )

            all_scenarios.append({
                'scenario_id': scen_idx,
                'temperatures': scenario_results,
            })

        # ── Cross-scenario summary ─────────────────────────────────────────
        summary = _summarise_rollout_results(all_scenarios, temperatures, feature_idx)

        out_data = {
            'feature_idx':  feature_idx,
            'temperatures': temperatures,
            'n_scenarios':  len(all_scenarios),
            'max_steps':    max_steps,
            'summary':      summary,
            'scenarios':    all_scenarios,
        }

        with open(output_path, 'w') as f:
            json.dump(out_data, f, indent=2, default=_json_default)

        print(f"\n[RolloutSteerer] Results → {output_path}")
        _print_summary_table(summary, temperatures)

        # ── Plots ─────────────────────────────────────────────────────────
        plot_rollout_metrics(summary, temperatures, feature_idx, out_dir)
        plot_action_trajectories(all_scenarios, temperatures, feature_idx, out_dir)

        return out_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _summarise_rollout_results(
    all_scenarios: List[Dict],
    temperatures: List[float],
    feature_idx: int,
) -> Dict:
    """Build cross-scenario aggregated statistics keyed by temperature."""
    summary: Dict[str, Any] = {}

    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        delta_lists: Dict[str, List[float]] = {}

        for scen in all_scenarios:
            for temp_result in scen['temperatures']:
                if temp_result['alpha'] != alpha:
                    continue
                for metric_name, dv in temp_result['delta_metrics'].items():
                    delta_lists.setdefault(metric_name, []).append(dv)

        agg: Dict[str, Any] = {}
        for metric_name, vals in delta_lists.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr) == 0:
                continue
            agg[metric_name] = {
                'mean':   float(arr.mean()),
                'std':    float(arr.std()),
                'median': float(np.median(arr)),
                'p5':     float(np.percentile(arr, 5)),
                'p95':    float(np.percentile(arr, 95)),
                'sign_flip_frac': float(
                    min((arr > 0).sum(), (arr < 0).sum()) / len(arr)
                ),
            }
        summary[key] = {'alpha': alpha, 'metrics': agg}

    return summary


def _print_summary_table(summary: Dict, temperatures: List[float]) -> None:
    """Print a compact ASCII table of mean Δmetric per temperature."""
    # Collect all metric names
    metric_names: List[str] = []
    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        if key in summary:
            metric_names = list(summary[key]['metrics'].keys())
            break

    if not metric_names:
        return

    col_w = 10
    header = f"{'metric':<30}" + "".join(f"{a:>{col_w}.2f}" for a in temperatures)
    print("\n" + header)
    print("-" * len(header))

    for mn in metric_names:
        row = f"{mn:<30}"
        for alpha in temperatures:
            key = f"alpha_{alpha:+.2f}"
            val = summary.get(key, {}).get('metrics', {}).get(mn, {}).get('mean', float('nan'))
            row += f"{val:>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rollout_metrics(
    summary: Dict,
    temperatures: List[float],
    feature_idx: int,
    out_dir: Path,
) -> None:
    """Bar chart of mean Δmetric per temperature for each vmax metric."""
    # Collect metric names
    metric_names: List[str] = []
    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        if key in summary:
            metric_names = list(summary[key]['metrics'].keys())
            break

    if not metric_names:
        return

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors_pos = '#4C72B0'
    colors_neg = '#DD8452'

    for ax, mn in zip(axes, metric_names):
        means = []
        stds  = []
        for alpha in temperatures:
            key = f"alpha_{alpha:+.2f}"
            stats = summary.get(key, {}).get('metrics', {}).get(mn, {})
            means.append(stats.get('mean', float('nan')))
            stds.append(stats.get('std', 0.0))

        means_arr = np.array(means)
        stds_arr  = np.array(stds)
        bar_colors = [colors_pos if m >= 0 else colors_neg for m in means_arr]

        ax.bar(range(len(temperatures)), means_arr, yerr=stds_arr,
               color=bar_colors, capsize=4, edgecolor='white', alpha=0.85)
        ax.axhline(0, color='crimson', linewidth=1.0, linestyle='--')
        ax.set_xticks(range(len(temperatures)))
        ax.set_xticklabels([f"{a:+.1f}" for a in temperatures], fontsize=8, rotation=45)
        ax.set_title(mn, fontsize=9, fontweight='bold')
        ax.set_xlabel('α', fontsize=8)
        ax.set_ylabel('Δ (steered − baseline)', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f'Feature {feature_idx} — Δmetric per temperature (paired rollouts)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out_path = out_dir / f'f{feature_idx}_rollout_metrics.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[RolloutSteerer] Plot → {out_path}")


def plot_action_trajectories(
    all_scenarios: List[Dict],
    temperatures: List[float],
    feature_idx: int,
    out_dir: Path,
) -> None:
    """For each temperature, plot baseline vs steered accel and steer over time.

    One figure per temperature.  Each figure has 2 rows (accel, steer) ×
    n_scenarios columns.
    """
    n_scen = len(all_scenarios)
    if n_scen == 0:
        return

    for alpha in temperatures:
        fig, axes = plt.subplots(
            2, n_scen, figsize=(4 * n_scen, 6), sharey='row', squeeze=False
        )

        for si, scen in enumerate(all_scenarios):
            # Find result for this alpha
            temp_result = next(
                (r for r in scen['temperatures'] if r['alpha'] == alpha), None
            )
            if temp_result is None:
                continue

            for dim_idx, dim_name in enumerate(['accel', 'steer']):
                ax = axes[dim_idx, si]

                b_vals = [s.get(dim_name, float('nan'))
                          for s in temp_result['per_step_baseline']]
                s_vals = [s.get(dim_name, float('nan'))
                          for s in temp_result['per_step_steered']]

                b_t = list(range(len(b_vals)))
                s_t = list(range(len(s_vals)))

                ax.plot(b_t, b_vals, color='#4C72B0', linewidth=1.4,
                        label='baseline', alpha=0.9)
                ax.plot(s_t, s_vals, color='#DD8452', linewidth=1.4,
                        linestyle='--', label=f'steered α={alpha:+.1f}', alpha=0.9)
                ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')

                if si == 0:
                    ax.set_ylabel(dim_name, fontsize=9)
                if dim_idx == 0:
                    ax.set_title(f'Scenario {si}', fontsize=9)
                if dim_idx == 1:
                    ax.set_xlabel('timestep', fontsize=8)
                if si == 0 and dim_idx == 0:
                    ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

        fig.suptitle(
            f'Feature {feature_idx} — Action trajectories  α={alpha:+.1f}',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        alpha_str = f"{alpha:+.1f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
        out_path = out_dir / f'f{feature_idx}_a{alpha_str}_trajectories.png'
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"[RolloutSteerer] Plot → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-timestep causal steering with vmax rollout metrics."
    )
    parser.add_argument("--run_dir",         required=True)
    parser.add_argument("--dataset",         required=True)
    parser.add_argument("--sae_checkpoint",  required=True)
    parser.add_argument("--feature_idx",     type=int, required=True)
    parser.add_argument(
        "--temperatures", type=float, nargs="+",
        default=[-5.0, -2.0, -1.0, 1.0, 2.0, 5.0],
    )
    parser.add_argument("--n_scenarios",  type=int, default=5)
    parser.add_argument("--max_steps",    type=int, default=80)
    parser.add_argument(
        "--output",
        default="data/sae_interpretability/rollout_results.json",
    )
    parser.add_argument("--checkpoint",   default="model_final.pkl")
    args = parser.parse_args()

    cfg = SAEConfig()
    steerer = RolloutSteerer(
        args.run_dir, args.dataset, args.sae_checkpoint, cfg, args.checkpoint
    )
    steerer.setup()
    steerer.run_rollout_experiment(
        feature_idx=args.feature_idx,
        temperatures=args.temperatures,
        n_scenarios=args.n_scenarios,
        max_steps=args.max_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
