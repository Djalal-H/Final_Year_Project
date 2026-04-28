# Copyright 2025 Valeo.

"""Feature annotation via z-score correlation with telemetry.

For each SAE latent dimension j (out of F = D * expansion_factor):
1. Find the top-K timesteps where feature j activates most strongly.
2. Query the aligned telemetry for those timesteps.
3. Compute z-scores: z = (mean_topK - mean_global) / std_global.
4. Rank telemetry fields by |z| to identify the concept the feature encodes.

Outputs a JSON file with per-feature annotations and a summary.

Usage:
    python -m xai.sae_interpretability.feature_annotator \\
        --data harvest.h5 \\
        --sae_checkpoint sae_model.pt \\
        --top_k 50 \\
        --output annotations.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from sae_interpretability.sae_model import SparseAutoencoder


# Which HDF5 telemetry fields to include in z-score analysis.
# Per-agent fields are collapsed to scalar summary statistics first.
_SCALAR_TEL_KEYS = ['ego_speed', 'ego_x', 'ego_y', 'min_ttc', 'min_agent_dist']
_BOOL_TEL_KEYS = ['is_ttc_critical', 'is_lead_vehicle_hard_braking']
_AGENT_KEYS_TO_SUMMARIZE = [
    ('agent_dists', ['mean', 'min']),
    ('agent_ttcs', ['mean', 'min']),
    ('agent_speeds', ['mean', 'max']),
]


def _load_telemetry(hf: h5py.File) -> Dict[str, np.ndarray]:
    """Load and flatten telemetry from an open HDF5 file."""
    tel: Dict[str, np.ndarray] = {}

    for k in _SCALAR_TEL_KEYS:
        path = f'telemetry/{k}'
        if path in hf:
            tel[k] = hf[path][:].astype(np.float32)

    for k in _BOOL_TEL_KEYS:
        path = f'telemetry/{k}'
        if path in hf:
            tel[k] = hf[path][:].astype(np.float32)  # float for z-score arithmetic

    for k, stats in _AGENT_KEYS_TO_SUMMARIZE:
        path = f'telemetry/{k}'
        if path not in hf:
            continue
        arr = hf[path][:]  # [N, n_agents]
        if 'mean' in stats:
            tel[f'{k}_mean'] = arr.mean(axis=1)
        if 'min' in stats:
            tel[f'{k}_min'] = arr.min(axis=1)
        if 'max' in stats:
            tel[f'{k}_max'] = arr.max(axis=1)

    return tel


def annotate(
    data_path: str,
    sae_checkpoint: str,
    top_k: int = 50,
    output_path: str = "annotations.json",
    min_activations: int = 10,
) -> List[Dict[str, Any]]:
    """Annotate SAE features by correlating with telemetry z-scores.

    Args:
        data_path: Path to harvest.h5.
        sae_checkpoint: Path to trained SAE .pt checkpoint.
        top_k: Number of top-activating timesteps per feature.
        output_path: Where to save the annotations JSON.
        min_activations: Features with fewer non-zero activations are labelled 'dead'.

    Returns:
        Sorted list of annotation dicts (active features first, then dead).
    """
    model = SparseAutoencoder.from_checkpoint(sae_checkpoint, map_location='cpu')
    model.eval()
    print(f"[Annotator] SAE: {model.input_dim}D → {model.latent_dim}D  ({sae_checkpoint})")

    print(f"[Annotator] Loading data from {data_path}")
    with h5py.File(data_path, 'r') as hf:
        activations = hf['activations'][:]   # [N, D]
        tel = _load_telemetry(hf)

    N, D = activations.shape
    print(f"[Annotator] {N:,} rows, {D} dims, {len(tel)} telemetry fields")

    # Apply the same mean-centering used during training
    act_mean: Optional[np.ndarray] = None
    try:
        ckpt = torch.load(sae_checkpoint, map_location='cpu', weights_only=False)
        act_mean = ckpt.get('act_mean', None)
    except Exception:
        pass

    act_in = (activations - act_mean).astype(np.float32) if act_mean is not None else activations.astype(np.float32)

    # Encode in batches to avoid OOM
    x = torch.from_numpy(act_in)
    chunks = []
    with torch.no_grad():
        for i in range(0, N, 8192):
            chunks.append(model.encode(x[i:i + 8192]).numpy())
    features = np.concatenate(chunks, axis=0)  # [N, F]

    F_dim = features.shape[1]
    print(f"[Annotator] Encoded → {features.shape}. Annotating {F_dim} features...")

    # Global stats for each telemetry field
    global_stats = {k: (float(v.mean()), float(v.std()) + 1e-8) for k, v in tel.items()}

    annotations: List[Dict[str, Any]] = []
    n_active = 0

    for j in range(F_dim):
        feat_j = features[:, j]
        n_nonzero = int((feat_j > 0).sum())

        if n_nonzero < min_activations:
            annotations.append({'feature_idx': j, 'label': 'dead', 'n_activations': n_nonzero})
            continue

        n_active += 1
        k_actual = min(top_k, n_nonzero)
        top_idx = np.argpartition(feat_j, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(feat_j[top_idx])[::-1]]

        z_scores: Dict[str, float] = {}
        for k, vals in tel.items():
            topk_mean = float(vals[top_idx].mean())
            g_mean, g_std = global_stats[k]
            z_scores[k] = round((topk_mean - g_mean) / g_std, 4)

        ranked = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_field, top_z = ranked[0]
        direction = "high" if top_z > 0 else "low"
        label = f"{direction}_{top_field} (z={top_z:+.2f})"

        annotations.append({
            'feature_idx': j,
            'label': label,
            'n_activations': n_nonzero,
            'mean_activation_when_active': round(float(feat_j[feat_j > 0].mean()), 5),
            'top_correlated_fields': [{'field': f, 'z': z} for f, z in ranked[:5]],
            'all_z_scores': z_scores,
        })

    print(f"[Annotator] {n_active}/{F_dim} features active (≥{min_activations} activations)")

    active = sorted(
        [a for a in annotations if a['label'] != 'dead'],
        key=lambda a: a['n_activations'],
        reverse=True,
    )
    dead = [a for a in annotations if a['label'] == 'dead']
    sorted_annotations = active + dead

    summary = {
        'total_features': F_dim,
        'active_features': n_active,
        'dead_features': F_dim - n_active,
        'dead_pct': round(100.0 * (F_dim - n_active) / F_dim, 2),
        'top_10_features': [
            {'idx': a['feature_idx'], 'label': a['label'], 'n_activations': a['n_activations']}
            for a in active[:10]
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'summary': summary, 'annotations': sorted_annotations}, f, indent=2)

    print(f"[Annotator] Saved → {output_path}")
    _print_top_features(active[:10])
    return sorted_annotations


def _print_top_features(features: List[Dict[str, Any]]) -> None:
    print("\n--- Top 10 Most Active Features ---")
    for ann in features:
        print(f"  [{ann['feature_idx']:4d}] n={ann['n_activations']:5d}  {ann['label']}")
        for entry in ann.get('top_correlated_fields', [])[:3]:
            print(f"           {entry['field']}: z={entry['z']:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Annotate SAE features via telemetry z-scores.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", default="annotations.json")
    parser.add_argument("--min_activations", type=int, default=10)
    args = parser.parse_args()

    annotate(args.data, args.sae_checkpoint, args.top_k, args.output, args.min_activations)


if __name__ == "__main__":
    main()
