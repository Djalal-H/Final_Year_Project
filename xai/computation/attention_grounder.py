# Copyright 2025 - Attention Grounder for Attention-Grounded XRL Pipeline
"""
Module 6: AttentionGrounder

Given raw cross-attention weights from the LQ/Perceiver encoder and the
threat agents identified by the NecessityScorer, compute the Attention
Grounding Score — a measure of whether the model was actually attending
to the agents that would have caused failures.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def compute(
    attention_weights: Dict[str, Any],
    encoder_layout: Dict[str, Any],
    threat_agents: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute the attention grounding score.

    Args:
        attention_weights: Dict from LQEncoder. Expected key (from config):
            e.g. ``cross_attn_0`` → array of shape ``[B, N_heads, N_latents, N_tokens]``
            or ``[N_heads, N_latents, N_tokens]`` (single sample).
        encoder_layout: Dict with ``n_sdc_timesteps``, ``num_objects``,
            ``timestep_agent``, ``roadgraph_top_k``, ``num_traffic_lights``,
            ``tl_timesteps``, ``gps_path_len``.
        threat_agents: List of ``{agent_id, min_ttc}`` dicts from NecessityScorer.
        config: Pipeline config dict (needs ``attention_layer_key``).

    Returns:
        Dictionary with:
            - ``grounding_score``: float or None (if no threat agents)
            - ``per_agent_breakdown``: list of ``{agent_id, severity, attention_mass}``
    """
    if not threat_agents:
        return {"grounding_score": None, "per_agent_breakdown": []}

    # 1. Get the cross-attention weights for the target layer
    layer_key = config.get("attention_layer_key", "cross_attn_0")
    attn = attention_weights.get(layer_key)
    if attn is None:
        return {"grounding_score": None, "per_agent_breakdown": []}

    attn = np.asarray(attn)

    # Handle batch dimension: use first sample if batched
    # Expected: [B, N_heads, N_latents, N_tokens] or [N_heads, N_latents, N_tokens]
    if attn.ndim == 4:
        attn = attn[0]  # → [N_heads, N_latents, N_tokens]

    # 2. Aggregate across heads and latents → per-token attention mass
    # Sum over heads (axis 0) and latents (axis 1) → [N_tokens]
    per_token_mass = attn.sum(axis=(0, 1))

    # 3. Compute token boundaries for the vehicle region
    n_sdc = encoder_layout.get("n_sdc_timesteps", 11)  # SDC tokens
    num_objects = encoder_layout.get("num_objects", 64)
    timestep_agent = encoder_layout.get("timestep_agent", 11)

    vehicles_start = n_sdc
    vehicles_end = vehicles_start + num_objects * timestep_agent

    # 4. Compute per-agent attention mass
    #    Agent j occupies tokens [vehicles_start + j*T .. vehicles_start + (j+1)*T)
    per_agent_mass = np.zeros(num_objects, dtype=np.float64)
    for j in range(num_objects):
        start = vehicles_start + j * timestep_agent
        end = start + timestep_agent
        if end <= len(per_token_mass):
            per_agent_mass[j] = per_token_mass[start:end].sum()

    # 5. Normalize to sum to 1 across all agents
    total = per_agent_mass.sum()
    if total > 0:
        per_agent_mass = per_agent_mass / total

    # 6. Compute severity and grounding score
    breakdown = []
    grounding_score = 0.0

    for ta in threat_agents:
        aid = ta["agent_id"]
        min_ttc = ta["min_ttc"]
        severity = 1.0 / (1.0 + min_ttc)

        if 0 <= aid < num_objects:
            a_mass = float(per_agent_mass[aid])
        else:
            a_mass = 0.0

        grounding_score += a_mass * severity
        breakdown.append(
            {
                "agent_id": aid,
                "severity": round(severity, 4),
                "attention_mass": round(a_mass, 6),
            }
        )

    return {
        "grounding_score": round(float(grounding_score), 4),
        "per_agent_breakdown": breakdown,
    }
