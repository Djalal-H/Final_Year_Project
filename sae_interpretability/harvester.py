# Copyright 2025 Valeo.

"""Activation harvester for SAE training data collection.

Extends OfflineExtractor to capture residual stream vectors and rich telemetry
in HDF5 format. The harvested data is the input to sae_trainer.py.

Usage:
    python -m sae_interpretability.harvester \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --n_scenarios 500 --output harvest.h5
"""

from __future__ import annotations
from sae_interpretability.config import SAEConfig
from xai.attention_analysis.offline_extraction import OfflineExtractor
import jax.numpy as jnp
import jax
import h5py

import argparse
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Scalar telemetry fields written as float32 [N]
_SCALAR_KEYS = ['ego_speed', 'ego_x', 'ego_y', 'min_ttc', 'min_agent_dist']

# Boolean telemetry fields written as bool [N]
_BOOL_KEYS = ['is_ttc_critical', 'is_lead_vehicle_hard_braking']

# Per-agent fields written as float32 [N, n_agents]
_AGENT_FLOAT_KEYS = ['agent_dists', 'agent_ttcs', 'agent_speeds',
                     'agent_is_ahead', 'agent_is_left', 'agent_closing_speed']


class ActivationHarvester(OfflineExtractor):
    """Harvests residual stream activations and extended telemetry for SAE training.

    Extends OfflineExtractor to:
    1. Capture the encoder output (residual stream) at every rollout timestep.
    2. Compute two boolean event flags that are easy to correlate with SAE features:
       - is_ttc_critical: True when min TTC across valid agents < 1.5 s.
       - is_lead_vehicle_hard_braking: True when the closest ahead-agent
         decelerates > 0.4g in a single 0.1 s step.
    3. Write everything to a resizable HDF5 file suitable for PyTorch DataLoader.
    """

    G = 9.81  # m/s²

    def __init__(
        self,
        run_dir: str,
        dataset_path: str,
        cfg: SAEConfig,
        checkpoint_name: str = "model_final.pkl",
    ):
        super().__init__(run_dir, dataset_path, checkpoint_name)
        self.sae_cfg = cfg

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _compute_lead_vehicle_hard_braking(
        self,
        curr_sem: Dict[str, np.ndarray],
        prev_sem: Optional[Dict[str, np.ndarray]],
    ) -> bool:
        """True if the closest ahead-agent drops speed by > 0.4g * dt in one step."""
        if prev_sem is None:
            return False

        is_ahead = np.asarray(curr_sem.get('is_ahead', []), dtype=bool)
        valid = np.asarray(curr_sem.get('valid', []), dtype=bool)
        distances = np.asarray(curr_sem.get('distance_to_ego', []))
        curr_speeds = np.asarray(curr_sem.get('agent_speeds', []))
        prev_speeds = np.asarray(prev_sem.get(
            'agent_speeds', np.zeros_like(curr_speeds)))

        n = min(len(is_ahead), len(valid), len(distances),
                len(curr_speeds), len(prev_speeds))
        if n == 0:
            return False

        is_ahead = is_ahead[:n]
        valid = valid[:n]
        distances = distances[:n]
        curr_speeds = curr_speeds[:n]
        prev_speeds = prev_speeds[:n]

        ahead_and_valid = is_ahead & valid
        if not ahead_and_valid.any():
            return False

        ahead_dists = np.where(ahead_and_valid, distances, np.inf)
        lead_idx = int(np.argmin(ahead_dists))

        threshold = self.sae_cfg.hard_braking_g_threshold * \
            self.G * self.sae_cfg.harvest_dt
        speed_drop = float(prev_speeds[lead_idx]) - \
            float(curr_speeds[lead_idx])
        return bool(speed_drop > threshold)

    def _build_telemetry_row(
        self,
        semantic: Dict[str, np.ndarray],
        prev_semantic: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """Flatten a semantic features dict into a single telemetry row."""
        row: Dict[str, Any] = {}

        sdc_idx = int(semantic.get('sdc_index', 0))

        # Ego state
        row['ego_speed'] = float(semantic.get('ego_speed', 0.0))
        pos_x = np.asarray(semantic.get('positions_x', [0.0]))
        pos_y = np.asarray(semantic.get('positions_y', [0.0]))
        row['ego_x'] = float(pos_x[sdc_idx]) if sdc_idx < len(pos_x) else 0.0
        row['ego_y'] = float(pos_y[sdc_idx]) if sdc_idx < len(pos_y) else 0.0

        # Per-agent arrays
        distances = np.asarray(semantic.get('distance_to_ego', []))
        ttc = np.asarray(semantic.get('ttc', []))
        speeds = np.asarray(semantic.get('agent_speeds', []))
        valid = np.asarray(semantic.get('valid', []), dtype=bool)
        closing = np.asarray(semantic.get('closing_speed', []))
        is_ahead = np.asarray(semantic.get('is_ahead', []))
        is_left = np.asarray(semantic.get('is_left', []))

        row['agent_dists'] = distances
        row['agent_ttcs'] = ttc
        row['agent_speeds'] = speeds
        row['agent_valid'] = valid
        row['agent_closing_speed'] = closing
        row['agent_is_ahead'] = is_ahead
        row['agent_is_left'] = is_left

        # Scalar summaries
        valid_ttc = ttc[valid] if (
            len(ttc) > 0 and len(valid) > 0) else np.array([])
        min_ttc = float(np.min(valid_ttc)) if len(valid_ttc) > 0 else 5.0
        min_dist = float(np.min(distances[valid])) if valid.any() and len(
            distances) > 0 else 999.0

        row['min_ttc'] = min_ttc
        row['min_agent_dist'] = min_dist

        # Boolean event flags
        row['is_ttc_critical'] = bool(
            min_ttc < self.sae_cfg.ttc_critical_threshold)
        row['is_lead_vehicle_hard_braking'] = self._compute_lead_vehicle_hard_braking(
            semantic, prev_semantic
        )

        return row

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_and_write(ds: h5py.Dataset, start: int, data: np.ndarray):
        new_size = start + len(data)
        ds.resize(new_size, axis=0)
        ds[start:new_size] = data

    # ------------------------------------------------------------------
    # Main harvesting entry point
    # ------------------------------------------------------------------

    def run_harvest(self, n_scenarios: int, output_path: str) -> None:
        """Process n_scenarios and write activations + telemetry to HDF5.

        Args:
            n_scenarios: Number of scenarios to process.
            output_path: Destination HDF5 file path.
        """
        from vmax.simulator import make_data_generator, datasets

        if self.encoder is None:
            self.setup()

        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

        @jax.jit
        def _forward_latent(e_params, obs):
            latent, _ = self.encoder.apply({'params': e_params}, obs)
            return latent

        os.makedirs(os.path.dirname(os.path.abspath(output_path))
                    or '.', exist_ok=True)
        D = self.sae_cfg.wayformer_hidden_dim
        total_rows = 0

        with h5py.File(output_path, 'w') as hf:
            # Resizable activation + index datasets
            act_ds = hf.create_dataset('activations', shape=(0, D), maxshape=(None, D),
                                       dtype='float32', chunks=(4096, D))
            sid_ds = hf.create_dataset('scenario_ids', shape=(0,), maxshape=(None,),
                                       dtype='int32', chunks=(4096,))
            ts_ds = hf.create_dataset('timesteps', shape=(0,), maxshape=(None,),
                                      dtype='int32', chunks=(4096,))

            # Scalar / bool telemetry datasets (pre-created)
            scalar_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,), maxshape=(None,),
                                     dtype='float32', chunks=(4096,))
                for k in _SCALAR_KEYS
            }
            bool_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,), maxshape=(None,),
                                     dtype='bool', chunks=(4096,))
                for k in _BOOL_KEYS
            }
            # Per-agent datasets created lazily on first batch (need n_agents)
            agent_float_ds: Dict[str, h5py.Dataset] = {}
            agent_valid_ds: Optional[h5py.Dataset] = None

            print(
                f"[Harvester] Processing {n_scenarios} scenarios → {output_path}")

            for scenario_idx, scenario_batch in enumerate(data_gen):
                if scenario_idx >= n_scenarios:
                    break

                print(
                    f"  Scenario {scenario_idx + 1}/{n_scenarios}", end="", flush=True)

                try:
                    env_transition = self.env.reset(scenario_batch)
                    batch_latents: List[np.ndarray] = []
                    batch_tel: List[Dict] = []
                    batch_ts: List[int] = []
                    prev_semantic = None

                    for t in range(self.sae_cfg.harvest_max_timesteps):
                        obs = env_transition.observation
                        latent = _forward_latent(self.encoder_params, obs)
                        latent_np = np.array(jax.device_get(latent))
                        if latent_np.ndim > 1:
                            latent_np = latent_np[0]  # squeeze batch dim → [D]

                        squeezed_state = jax.tree_util.tree_map(
                            lambda x: x.squeeze(0) if hasattr(
                                x, 'squeeze') and x.ndim > 0 else x,
                            env_transition.state,
                        )
                        semantic = self.extract_semantic_features(
                            squeezed_state)
                        tel_row = self._build_telemetry_row(
                            semantic, prev_semantic)

                        batch_latents.append(latent_np)
                        batch_tel.append(tel_row)
                        batch_ts.append(t)
                        prev_semantic = semantic

                        if t > 0 and bool(jax.device_get(env_transition.done)):
                            break

                        try:
                            env_transition = self._expert_step_single(
                                env_transition)
                        except Exception:
                            break

                    if not batch_latents:
                        print(" ✗ empty")
                        continue

                    n_new = len(batch_latents)
                    act_arr = np.stack(batch_latents, axis=0)  # [T, D]

                    self._resize_and_write(act_ds, total_rows, act_arr)
                    self._resize_and_write(
                        sid_ds, total_rows,
                        np.full(n_new, scenario_idx, dtype='int32')
                    )
                    self._resize_and_write(
                        ts_ds, total_rows,
                        np.array(batch_ts, dtype='int32')
                    )

                    for k in _SCALAR_KEYS:
                        self._resize_and_write(
                            scalar_ds[k], total_rows,
                            np.array([r[k]
                                     for r in batch_tel], dtype='float32')
                        )
                    for k in _BOOL_KEYS:
                        self._resize_and_write(
                            bool_ds[k], total_rows,
                            np.array([r[k] for r in batch_tel], dtype=bool)
                        )

                    # Per-agent datasets — create on first batch
                    sample = batch_tel[0]
                    n_agents = len(np.asarray(sample.get('agent_dists', [])))

                    for k in _AGENT_FLOAT_KEYS:
                        if k not in agent_float_ds:
                            agent_float_ds[k] = hf.create_dataset(
                                f'telemetry/{k}', shape=(0, n_agents),
                                maxshape=(None, n_agents), dtype='float32',
                                chunks=(4096, n_agents)
                            )
                        mat = np.array([
                            np.asarray(r[k], dtype='float32')[:n_agents]
                            for r in batch_tel
                        ])
                        self._resize_and_write(
                            agent_float_ds[k], total_rows, mat)

                    if agent_valid_ds is None:
                        agent_valid_ds = hf.create_dataset(
                            'telemetry/agent_valid', shape=(0, n_agents),
                            maxshape=(None, n_agents), dtype='bool',
                            chunks=(4096, n_agents)
                        )
                    mat_valid = np.array([
                        np.asarray(r['agent_valid'], dtype=bool)[:n_agents]
                        for r in batch_tel
                    ])
                    self._resize_and_write(
                        agent_valid_ds, total_rows, mat_valid)

                    total_rows += n_new
                    print(f" ✓ ({n_new} steps, total={total_rows})")

                except Exception as e:
                    print(f" ✗ {e}")
                    traceback.print_exc()

            # Metadata
            meta = hf.create_group('metadata')
            meta.attrs['checkpoint'] = self.checkpoint_name
            meta.attrs['run_dir'] = self.run_dir
            meta.attrs['n_scenarios_requested'] = n_scenarios
            meta.attrs['n_rows'] = total_rows
            meta.attrs['hidden_dim'] = D
            meta.attrs['sae_expansion_factor'] = self.sae_cfg.sae_expansion_factor
            meta.attrs['ttc_critical_threshold'] = self.sae_cfg.ttc_critical_threshold
            meta.attrs['hard_braking_g_threshold'] = self.sae_cfg.hard_braking_g_threshold

        print(f"\n[Harvester] Done. {total_rows} rows → {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Harvest Wayformer residual stream activations for SAE training."
    )
    parser.add_argument("--run_dir", required=True,
                        help="Path to training run directory")
    parser.add_argument("--dataset", required=True,
                        help="Dataset path or name")
    parser.add_argument("--n_scenarios", type=int, default=500)
    parser.add_argument("--output", default="harvest.h5")
    parser.add_argument("--checkpoint", default="model_final.pkl")
    parser.add_argument("--expansion_factor", type=int, default=16)
    args = parser.parse_args()

    cfg = SAEConfig(sae_expansion_factor=args.expansion_factor)
    harvester = ActivationHarvester(
        args.run_dir, args.dataset, cfg, args.checkpoint)
    harvester.setup()
    harvester.run_harvest(args.n_scenarios, args.output)


if __name__ == "__main__":
    main()
