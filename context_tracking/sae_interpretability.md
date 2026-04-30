# Phase Overview
This development phase, titled "Sparse Autoencoder (SAE) Interpretability Integration," focuses on establishing a robust, end-to-end pipeline for understanding the internal representations of the Wayformer model. The primary objective is to extract dense, continuous embeddings from the model's residual stream and map them into a sparse, interpretable latent space where individual features correspond to distinct, human-understandable driving concepts.

# Theoretical Foundation & Design Goals
Deep neural networks often exhibit "superposition," where the number of learned concepts exceeds the available mathematical dimensions, leading to polysemantic neurons that are difficult to interpret. To address this, we leverage Sparse Autoencoders (SAEs), which act as a decompressive lens. 

The core design paradigm involves training an auxiliary SAE on the frozen activations of the target model (Wayformer). By enforcing strict sparsity penalties, the SAE is coerced to represent the dense embeddings within a significantly wider latent space, ensuring that each active dimension (feature) represents a single, isolated concept (monosemanticity). This approach allows us to directly correlate active network features with specific simulation telemetry (e.g., time-to-collision, lead vehicle behavior) to gain granular insights into the model's decision-making process prior to the reinforcement learning policy step.

# Implementation Summary
The implementation successfully establishes the complete SAE lifecycle, comprising data harvesting, model definition, training, and configuration management:

*   **Activation Harvesting (`harvester.py`):** An optimized data extraction pipeline was implemented utilizing `jax.lax.scan`. This allows for the compilation of entire simulation episodes into highly efficient XLA kernels, dramatically increasing the throughput of capturing residual stream activations and complex telemetry metrics (distances, closing speeds, etc.) across multiple scenarios simultaneously.
*   **SAE Model Architecture (`sae_model.py`):** The core Sparse Autoencoder was constructed using PyTorch, adhering to established conventions (e.g., pre-encoder bias subtraction, unit-norm decoder constraints). A key addition is the implementation of a custom `JumpReLU` activation function.
*   **Training & Tuning Pipeline (`sae_trainer.py`, `sae_tuner.py`):** A comprehensive training loop was developed, incorporating data normalization (mean-centering and standard deviation whitening) to ensure stable convergence. Furthermore, an automated hyperparameter tuning script was integrated to systematically explore different architectural configurations and sparsity penalties via grid search.
*   **Configuration Management (`config.py`, `sae_config.yaml`):** The pipeline employs a centralized YAML configuration, loaded dynamically at runtime, establishing a single source of truth for all architectural and training parameters.

# Key Decisions & Rationale

*   **Utilization of `jax.lax.scan` for Harvesting:** 
    *   *Rationale:* Transitioning from Python-level loops to JAX-compiled scans for scenario rollouts significantly reduces I/O bottlenecks and context-switching overhead. This architectural choice maximizes hardware utilization during the data collection phase, allowing for the rapid generation of the large datasets required for effective SAE training.
*   **Integration of `JumpReLU` Activation:** 
    *   *Rationale:* Standard ReLU activations coupled with L1 regularization can sometimes lead to excessive feature shrinkage, where marginal features are incorrectly driven to zero. `JumpReLU` introduces a hard threshold gate with a straight-through estimator for the backward pass. This ensures that relevant features pass through with their full magnitude while effectively filtering noise, preventing feature collapse and promoting a healthier sparsity ratio (L0).
*   **Data Whitening (Mean & Std Normalization):** 
    *   *Rationale:* Standardizing the harvested activations before SAE encoding ensures that all latent dimensions contribute equally to the learning process. This addresses potential issues where a high-magnitude baseline or dominant dimensions might mask subtle, underlying features, leading to more robust and distinct concept discovery.
*   **Automated Grid Search Tuning:** 
    *   *Rationale:* Identifying the optimal balance between reconstruction accuracy and feature sparsity is highly empirical. Providing an automated tuner allows for the systematic evaluation of critical hyperparameters (such as the L1 coefficient and expansion factor), ensuring the final model achieves the desired feature isolation without manual trial and error.
*   **Centralized YAML Configuration:** 
    *   *Rationale:* Decoupling configuration from code execution ensures consistency across the harvesting, training, and tuning stages. It simplifies experimentation and guarantees that all pipeline components operate under identical assumptions.
