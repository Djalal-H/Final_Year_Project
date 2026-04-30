# Copyright 2025 Valeo.

"""Causal steering: modify a single SAE feature and measure behavioral change.

Intervention protocol for feature j at temperature α:
    h  (residual stream)
    ↓  SAE encode
    f  = ReLU(h @ W_enc + b_enc)
    ↓  modify
    f' = f.at[j].add(α)
    ↓  SAE decode
    h_steered = f' @ W_dec + b_dec
    ↓  apply delta in residual stream space
    h_final = h + (h_steered - h_reconstructed)
    ↓  policy / value FC heads
    Δvalue, Δlogit_norm  ← behavioral effect

A large Δvalue that scales monotonically with α confirms that feature j
causally drives the agent's value estimate.

Usage:
    python -m sae_interpretability.causal_steering \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --sae_checkpoint sae_model.pt \\
        --feature_idx 42 \\
        --temperatures 0.5 1.0 2.0 5.0
"""

from __future__ import annotations
from sae_interpretability.sae_model import SparseAutoencoder
from sae_interpretability.config import SAEConfig
from xai.attention_analysis.offline_extraction import OfflineExtractor

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CausalSteerer(OfflineExtractor):
    """Apply causal interventions on SAE features and measure behavioral change.

    Inherits env setup, model loading, and data iteration from OfflineExtractor.
    Adds SAE weight loading and a steering-specific forward pass.
    """

    def __init__(
        self,
        run_dir: str,
        dataset_path: str,
        sae_checkpoint: str,
        cfg: SAEConfig,
        checkpoint_name: str = "model_final.pkl",
    ):
        super().__init__(run_dir, dataset_path, checkpoint_name)
        self.sae_checkpoint = sae_checkpoint
        self.sae_cfg = cfg
        self._sae_w: Optional[Dict[str, jnp.ndarray]] = None
        self._act_mean_jnp: Optional[jnp.ndarray] = None
        self._act_std_jnp:  Optional[jnp.ndarray] = None
        self._policy_fc_params = None
        self._policy_dense_params = None
        self._value_fc_params = None
        self._value_dense_params = None

    def setup(self) -> CausalSteerer:
        super().setup()
        self._load_sae_weights()
        self._extract_head_params()
        return self

    def _load_sae_weights(self):
        """Load the trained SAE checkpoint and convert weights to JAX arrays."""
        print(f"[Steerer] Loading SAE from {self.sae_checkpoint}")
        ckpt = torch.load(self.sae_checkpoint,
                          map_location='cpu', weights_only=False)
        sae = SparseAutoencoder(
            input_dim=ckpt['config']['input_dim'],
            latent_dim=ckpt['config']['latent_dim'],
            l1_coeff=ckpt['config']['l1_coeff'],
        )
        sae.load_state_dict(ckpt['model_state_dict'])
        np_w = sae.to_numpy_weights()
        self._sae_w = {k: jnp.array(v) for k, v in np_w.items()}

        input_dim = ckpt['config']['input_dim']
        act_mean_np = ckpt.get('act_mean', np.zeros(input_dim, dtype=np.float32))
        act_std_np  = ckpt.get('act_std',  np.ones(input_dim,  dtype=np.float32))
        self._act_mean_jnp = jnp.array(act_mean_np.squeeze())
        self._act_std_jnp  = jnp.array(act_std_np.squeeze())
        print(
            f"[Steerer] SAE ready: "
            f"{ckpt['config']['input_dim']}D → {ckpt['config']['latent_dim']}D"
        )

    def _extract_head_params(self):
        """Pull policy and value FC-head params out of the loaded checkpoint."""
        policy_p = self.params.policy.get('params', self.params.policy)
        self._policy_fc_params = policy_p.get('fully_connected_layer')
        self._policy_dense_params = policy_p.get('Dense_0')

        if hasattr(self.params, 'value'):
            value_p = self.params.value.get('params', self.params.value)
            self._value_fc_params = value_p.get('fully_connected_layer')
            self._value_dense_params = value_p.get('Dense_0')

    # ------------------------------------------------------------------
    # SAE encode / decode  (pure JAX, JIT-able)
    # ------------------------------------------------------------------

    def _sae_encode(self, h: jnp.ndarray) -> jnp.ndarray:
        w = self._sae_w
        return jnp.maximum(((h - self._act_mean_jnp) / self._act_std_jnp) @ w['W_enc'] + w['b_enc'], 0.0)

    def _sae_decode(self, f: jnp.ndarray) -> jnp.ndarray:
        w = self._sae_w
        return f @ w['W_dec'] + w['b_dec']

    # ------------------------------------------------------------------
    # FC head application
    # ------------------------------------------------------------------

    def _apply_fc_head(
        self,
        h: jnp.ndarray,
        fc_params: Optional[Dict],
        dense_params: Optional[Dict],
    ) -> Optional[jnp.ndarray]:
        """Apply MLP + final Dense layer to a latent vector.

        Returns None when parameters are unavailable.
        """
        if fc_params is None or dense_params is None:
            return None
        x = h
        try:
            for key in sorted(fc_params.keys()):
                layer = fc_params[key]
                if 'kernel' in layer and 'bias' in layer:
                    x = jnp.maximum(x @ layer['kernel'] + layer['bias'], 0.0)
            x = x @ dense_params['kernel'] + dense_params['bias']
        except Exception as e:
            print(f"[Steerer] FC head error: {e}")
            return None
        return x

    # ------------------------------------------------------------------
    # Single-observation steering
    # ------------------------------------------------------------------

    def steer_observation(
        self,
        obs: jnp.ndarray,
        feature_idx: int,
        temperatures: List[float],
    ) -> Dict[str, Any]:
        """Measure the causal effect of steering feature j on a single observation.

        Args:
            obs: Vectorized observation [obs_dim] or [1, obs_dim].
            feature_idx: SAE feature index to modify.
            temperatures: List of α values to add to feature j.

        Returns:
            Dict containing baseline activation and per-temperature behavioral shift.
        """
        if obs.ndim == 1:
            obs = obs[None]  # → [1, obs_dim]

        h, _ = self.encoder.apply({'params': self.encoder_params}, obs)
        if h.ndim > 1:
            h = h[0]  # [D]

        f_baseline = self._sae_encode(h)
        h_reconstructed = self._sae_decode(f_baseline)  # baseline SAE output

        result: Dict[str, Any] = {
            'feature_idx': feature_idx,
            'baseline_activation': float(f_baseline[feature_idx]),
            'temperatures': [],
        }

        h_batch = h[None]  # [1, D] for FC heads
        policy_base = self._apply_fc_head(
            h_batch, self._policy_fc_params, self._policy_dense_params)
        value_base = self._apply_fc_head(
            h_batch, self._value_fc_params, self._value_dense_params)

        for alpha in temperatures:
            f_steered = f_baseline.at[feature_idx].add(alpha)
            h_steered = self._sae_decode(f_steered)

            # Apply only the delta so we stay in the residual stream manifold
            h_final = (h + (h_steered - h_reconstructed))[None]  # [1, D]

            entry: Dict[str, Any] = {
                'alpha': alpha,
                'steered_activation': float(f_steered[feature_idx]),
                'h_delta_norm': float(jnp.linalg.norm(h_steered - h_reconstructed)),
            }

            policy_steered = self._apply_fc_head(
                h_final, self._policy_fc_params, self._policy_dense_params
            )
            value_steered = self._apply_fc_head(
                h_final, self._value_fc_params, self._value_dense_params
            )

            if policy_base is not None and policy_steered is not None:
                entry['delta_policy_norm'] = float(
                    jnp.linalg.norm(policy_steered - policy_base)
                )
            if value_base is not None and value_steered is not None:
                entry['delta_value'] = float(
                    jnp.mean(value_steered - value_base))

            result['temperatures'].append(entry)

        return result

    # ------------------------------------------------------------------
    # Multi-scenario experiment
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        feature_idx: int,
        temperatures: List[float],
        n_scenarios: int = 10,
        output_path: str = "steering_results.json",
    ) -> Dict[str, Any]:
        """Run the steering experiment across multiple scenarios and aggregate.

        Args:
            feature_idx: SAE feature to steer.
            temperatures: α values to try.
            n_scenarios: How many scenarios to evaluate.
            output_path: Where to write the JSON results.

        Returns:
            Dict with per-scenario results and aggregated statistics.
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

        all_results: List[Dict] = []
        print(
            f"[Steerer] Feature {feature_idx}  α={temperatures}  "
            f"{n_scenarios} scenarios"
        )

        for i, scenario_batch in enumerate(data_gen):
            if i >= n_scenarios:
                break
            env_transition = self.env.reset(scenario_batch)
            obs = env_transition.observation

            result = self.steer_observation(obs, feature_idx, temperatures)
            result['scenario_id'] = i
            all_results.append(result)

            baseline = result['baseline_activation']
            print(f"  Scenario {i+1:3d}: baseline_activation={baseline:.4f}")

        # Aggregate delta_value per temperature
        n_temps = len(temperatures)
        delta_matrix = np.zeros((len(all_results), n_temps))
        for si, res in enumerate(all_results):
            for ti, entry in enumerate(res['temperatures']):
                delta_matrix[si, ti] = entry.get('delta_value', 0.0)

        summary = {
            'feature_idx': feature_idx,
            'temperatures': temperatures,
            'mean_delta_value': delta_matrix.mean(axis=0).tolist(),
            'std_delta_value': delta_matrix.std(axis=0).tolist(),
            'mean_baseline_activation': float(
                np.mean([r['baseline_activation'] for r in all_results])
            ),
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path))
                    or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(
                {'summary': summary, 'per_scenario': all_results}, f, indent=2)

        print(f"\n[Steerer] Results → {output_path}")
        print("  Mean Δvalue per α:")
        for alpha, dv in zip(temperatures, summary['mean_delta_value']):
            bar = '▮' * int(abs(dv) * 20)
            sign = '+' if dv >= 0 else ''
            print(f"    α={alpha:5.1f}: {sign}{dv:.4f}  {bar}")

        return {'summary': summary, 'per_scenario': all_results}


def main():
    parser = argparse.ArgumentParser(
        description="Causal steering experiment on SAE features.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--feature_idx", type=int, required=True)
    parser.add_argument("--temperatures", type=float,
                        nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--n_scenarios", type=int, default=10)
    parser.add_argument("--output", default="steering_results.json")
    parser.add_argument("--checkpoint", default="model_final.pkl")
    args = parser.parse_args()

    cfg = SAEConfig()
    steerer = CausalSteerer(
        args.run_dir, args.dataset, args.sae_checkpoint, cfg, args.checkpoint
    )
    steerer.setup()
    steerer.run_experiment(
        args.feature_idx, args.temperatures, args.n_scenarios, args.output
    )


if __name__ == "__main__":
    main()
