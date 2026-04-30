# Copyright 2025 Valeo.

"""Pipeline-wide configuration for the SAE interpretability pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class SAEConfig:
    # Wayformer residual stream dimension (dk * ff_mult from encoder config)
    wayformer_hidden_dim: int = 128

    # SAE architecture
    sae_expansion_factor: int = 16
    sae_l1_coeff: float = 1e-3
    sae_learning_rate: float = 3e-4
    sae_batch_size: int = 4096
    sae_epochs: int = 10

    # Feature annotation
    top_k_activations: int = 50

    # Causal steering
    steering_temperatures: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])

    # Harvesting
    harvest_max_timesteps: int = 80
    harvest_n_scenarios: int = 500
    harvest_dt: float = 0.1  # Waymo dataset: 10 Hz

    # Telemetry thresholds
    ttc_critical_threshold: float = 1.5      # seconds
    hard_braking_g_threshold: float = 0.4   # g-force

    # SAE activation function
    jump_threshold: float = 0.001  # JumpReLU hard gate threshold θ

    @property
    def sae_latent_dim(self) -> int:
        return self.wayformer_hidden_dim * self.sae_expansion_factor

    @classmethod
    def from_yaml(cls, path: str) -> SAEConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})
