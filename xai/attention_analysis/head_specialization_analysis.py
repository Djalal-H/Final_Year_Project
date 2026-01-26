# Copyright 2025 Valeo.

"""Head Specialization Analysis for XAI.

This module analyzes attention head behavior to discover functional specialization
during training. It computes Head Specialization Index (HSI) scores and generates
visualizations to understand what each attention head has learned to focus on.

Key Concepts:
- HSI (Head Specialization Index): Measures how strongly a head correlates with 
  specific semantic features (TTC, distance, etc.)
- Primary Function: The semantic feature with strongest correlation for each head
- Head Function Registry: Maps heads to interpretable function labels

Data Sources:
    This analysis requires data from TWO directories:
    1. feature_dir: Training logs with semantic features (from AttentionLogger)
       Located at: runs/{model}/attention_logs/
    2. attention_dir: Offline extracted attention weights
       Located at: xai/attention_analysis/attention_extractions/

Usage:
    # Load and analyze attention logs from both directories
    analyzer = HeadSpecializationAnalyzer(
        feature_dir="runs/PPO_VEC_WAYFORMER/attention_logs",
        attention_dir="xai/attention_analysis/attention_extractions"
    )
    analyzer.load_data()
    results = analyzer.compute_hsi()
    analyzer.export_registry("head_functions.json")
    
    # Or via CLI:
    # python head_specialization_analysis.py \\
    #     --feature_dir runs/PPO_VEC_WAYFORMER/attention_logs \\
    #     --attention_dir xai/attention_analysis/attention_extractions \\
    #     --output_dir hsi_results
"""

import glob
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class AttentionLog:
    """Container for a single attention log entry."""
    step: int
    attention_weights: Dict[str, np.ndarray]
    semantic_features: Dict[str, np.ndarray]
    token_boundaries: Dict[str, Tuple[int, int]]
    config: Dict[str, Any]


@dataclass
class HSIResult:
    """Results from Head Specialization Index computation."""
    hsi_scores: np.ndarray  # Shape: (n_heads,)
    correlations: Dict[str, np.ndarray]  # Feature -> (n_heads,) correlations
    primary_features: List[str]  # Primary feature per head
    primary_correlations: np.ndarray  # Correlation value for primary feature
    head_labels: Dict[int, Dict[str, Any]]  # Head index -> label info


# ==============================================================================
# Feature Labeling Configuration
# ==============================================================================

# Mapping from feature names to human-readable head function labels
FEATURE_TO_LABEL = {
    'ttc': {
        'name': 'Safety Head',
        'description': 'Attends to vehicles with low time-to-collision (collision threats)',
        'expected_sign': 'negative',  # Low TTC = high attention
    },
    'distance_to_ego': {
        'name': 'Proximity Head',
        'description': 'Attends to nearby vehicles',
        'expected_sign': 'negative',  # Close distance = high attention
    },
    'closing_speed': {
        'name': 'Threat Assessment Head',
        'description': 'Attends to rapidly approaching vehicles',
        'expected_sign': 'positive',  # High closing speed = high attention
    },
    'is_ahead': {
        'name': 'Traffic Flow Head',
        'description': 'Attends to vehicles ahead in the direction of travel',
        'expected_sign': 'positive',
    },
    'is_left': {
        'name': 'Left Lane Head',
        'description': 'Attends to vehicles in the left lane',
        'expected_sign': 'positive',
    },
    'is_right': {
        'name': 'Right Lane Head',
        'description': 'Attends to vehicles in the right lane',
        'expected_sign': 'positive',
    },
    'is_behind': {
        'name': 'Rear Monitoring Head',
        'description': 'Attends to vehicles behind the ego vehicle',
        'expected_sign': 'positive',
    },
    'agent_speeds': {
        'name': 'Dynamic Object Head',
        'description': 'Attends to fast-moving agents',
        'expected_sign': 'variable',
    },
}

# Default label for heads without strong specialization
DEFAULT_LABEL = {
    'name': 'General Context Head',
    'description': 'Diffuse attention without strong feature specialization',
}


# ==============================================================================
# Attention Log Loader
# ==============================================================================

class AttentionLogLoader:
    """Load and merge attention logs from training and offline extraction.
    
    This loader requires data from TWO directories:
    1. feature_dir: Training logs with semantic features (from AttentionLogger)
    2. attention_dir: Offline extracted attention weights
    
    Files are matched by sorted order (assumes parallel naming convention).
    """
    
    def __init__(
        self,
        feature_dir: str,
        attention_dir: str,
        feature_pattern: str = "attention_log_*.pkl",
        attention_pattern: str = "attention_scenario_*.pkl"
    ):
        """
        Initialize the loader.
        
        Args:
            feature_dir: Directory containing training logs (semantic features).
            attention_dir: Directory containing offline extractions (attention weights).
            feature_pattern: Glob pattern for feature log files.
            attention_pattern: Glob pattern for attention log files.
        """
        self.feature_dir = feature_dir
        self.attention_dir = attention_dir
        self.feature_pattern = feature_pattern
        self.attention_pattern = attention_pattern
        self.logs: List[AttentionLog] = []
        
    def discover_files(self) -> Tuple[List[str], List[str]]:
        """Find log files in both directories."""
        feature_files = sorted(glob.glob(os.path.join(self.feature_dir, self.feature_pattern)))
        attention_files = sorted(glob.glob(os.path.join(self.attention_dir, self.attention_pattern)))
        
        if not feature_files:
            print(f"[AttentionLogLoader] Warning: No feature logs found in {self.feature_dir}")
        if not attention_files:
            print(f"[AttentionLogLoader] Warning: No attention logs found in {self.attention_dir}")
            
        return feature_files, attention_files
    
    def load(self, step_range: Optional[Tuple[int, int]] = None) -> List[AttentionLog]:
        """
        Load and merge logs from both directories.
        
        Assumes files correspond 1-to-1 when sorted by name.
        
        Args:
            step_range: Optional (start, end) tuple to filter by training step.
        
        Returns:
            List of AttentionLog objects with merged data.
        """
        feature_files, attention_files = self.discover_files()
        
        if not feature_files or not attention_files:
            raise FileNotFoundError(
                "Missing required log files. Need both:\n"
                f"- Training logs (semantic features) in {self.feature_dir}\n"
                f"- Extracted attention weights in {self.attention_dir}"
            )
            
        if len(feature_files) != len(attention_files):
            print(f"[AttentionLogLoader] Warning: Mismatch in file counts "
                  f"({len(feature_files)} features vs {len(attention_files)} attention). "
                  f"Will load intersection.")
            min_len = min(len(feature_files), len(attention_files))
            feature_files = feature_files[:min_len]
            attention_files = attention_files[:min_len]
        
        print(f"[AttentionLogLoader] Loading/merging {len(feature_files)} log pairs...")
        
        self.logs = []
        for feat_path, attn_path in zip(feature_files, attention_files):
            try:
                # Load features from training log
                with open(feat_path, 'rb') as f:
                    feat_data = pickle.load(f)
                
                # Load attention from extraction log
                with open(attn_path, 'rb') as f:
                    attn_data = pickle.load(f)
                
                # Merge data
                log = self._merge_logs(feat_data, attn_data, feat_path)
                
                # Apply step filter
                if step_range is not None:
                    if log.step < step_range[0] or log.step > step_range[1]:
                        continue
                
                self.logs.append(log)
                
            except Exception as e:
                print(f"[AttentionLogLoader] Warning: Could not load pair ({feat_path}, {attn_path}): {e}")
        
        print(f"[AttentionLogLoader] Successfully loaded {len(self.logs)} logs")
        return self.logs
    
    def _merge_logs(self, feat_data: Dict, attn_data: Dict, feat_path: str) -> AttentionLog:
        """Merge feature data and attention data into one log object."""
        # Primary metadata comes from feature/training log
        step = feat_data.get('step', 0)
        
        # If step missing, try to infer from filename
        if step == 0:
            step = self._extract_step_from_filename(os.path.basename(feat_path))
            
        return AttentionLog(
            step=step,
            attention_weights=attn_data['attention_weights'],  # Use extracted attention
            semantic_features=feat_data.get('semantic_features', {}),  # Use training features
            token_boundaries=feat_data.get('token_boundaries', {}),
            config=feat_data.get('config', {})
        )
    
    def _extract_step_from_filename(self, filename: str) -> int:
        """Extract step number from filename like 'attention_log_00001000.pkl'."""
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0


# ==============================================================================
# Head Specialization Analyzer
# ==============================================================================

class HeadSpecializationAnalyzer:
    """
    Analyze attention head specialization through correlation analysis.
    
    This class computes the Head Specialization Index (HSI) for each attention
    head by correlating attention patterns with semantic features.
    """
    
    # Features to analyze for specialization
    ANALYSIS_FEATURES = [
        'ttc',
        'distance_to_ego',
        'closing_speed',
        'is_ahead',
        'is_left',
        'is_right',
        'is_behind',
        'agent_speeds',
    ]
    
    # Minimum HSI threshold for "specialized" heads
    HSI_THRESHOLD = 0.3
    
    def __init__(
        self,
        feature_dir: Optional[str] = None,
        attention_dir: Optional[str] = None,
        logs: Optional[List[AttentionLog]] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            feature_dir: Directory containing training logs (semantic features).
            attention_dir: Directory containing offline extractions (attention weights).
            logs: Pre-loaded list of AttentionLog objects.
        """
        self.feature_dir = feature_dir
        self.attention_dir = attention_dir
        self._logs: List[AttentionLog] = logs or []
        self._result: Optional[HSIResult] = None
        
        # Aggregated data for analysis
        self._attention_per_vehicle: Optional[np.ndarray] = None  # (N, H, V)
        self._feature_arrays: Dict[str, np.ndarray] = {}  # Feature -> (N, V)
        
    def load_data(self, step_range: Optional[Tuple[int, int]] = None):
        """Load and merge attention logs from both directories."""
        if self.feature_dir is None or self.attention_dir is None:
            raise ValueError("Both feature_dir and attention_dir must be specified")
        
        loader = AttentionLogLoader(self.feature_dir, self.attention_dir)
        self._logs = loader.load(step_range)
        return self
    
    @property
    def logs(self) -> List[AttentionLog]:
        """Get loaded attention logs."""
        return self._logs
    
    @property
    def result(self) -> Optional[HSIResult]:
        """Get HSI computation result."""
        return self._result
    
    def aggregate_attention_by_vehicle(self) -> np.ndarray:
        """
        Aggregate attention weights per vehicle across all logs.
        
        For each log, extracts the cross-attention from latents to vehicle tokens
        and sums over tokens belonging to each vehicle.
        
        Returns:
            Array of shape (N_scenarios, N_heads, N_vehicles) with attention
            paid to each vehicle by each head.
        """
        if not self._logs:
            raise ValueError("No logs loaded. Call load_data() first.")
        
        all_attention = []
        
        for log in self._logs:
            attn = self._extract_vehicle_attention(log)
            if attn is not None:
                all_attention.append(attn)
        
        if not all_attention:
            raise ValueError("Could not extract vehicle attention from any logs")
        
        # Stack along scenario dimension
        # Each attn is (n_samples, n_heads, n_vehicles)
        self._attention_per_vehicle = np.concatenate(all_attention, axis=0)
        
        print(f"[Analyzer] Aggregated attention shape: {self._attention_per_vehicle.shape}")
        return self._attention_per_vehicle
    
    def _extract_vehicle_attention(self, log: AttentionLog) -> Optional[np.ndarray]:
        """
        Extract per-vehicle attention from a single log.
        
        Looks for cross-attention keys like 'other_traj/cross_attn_0' which
        contain attention from latents to vehicle trajectory tokens.
        
        Returns:
            Array of shape (n_samples, n_heads, n_vehicles) or None if not found.
        """
        attn_weights = log.attention_weights
        
        # Find the vehicle cross-attention key
        vehicle_key = None
        for key in attn_weights.keys():
            if 'other_traj' in key and 'cross_attn' in key:
                vehicle_key = key
                break
        
        if vehicle_key is None:
            print(f"[Analyzer] Warning: No vehicle attention found in step {log.step}")
            return None
        
        # Shape: (n_samples, n_latents, n_vehicle_tokens, n_heads)
        attn = attn_weights[vehicle_key]
        
        # Rearrange to (n_samples, n_heads, n_latents, n_vehicle_tokens)
        if attn.ndim == 4:
            attn = np.transpose(attn, (0, 3, 1, 2))
        
        n_samples, n_heads, n_latents, n_tokens = attn.shape
        
        # Get vehicle count and timesteps from config or infer from shape
        config = log.config
        n_vehicles = config.get('num_objects', 64)
        timesteps = n_tokens // n_vehicles if n_vehicles > 0 else 1
        
        # If tokens don't divide evenly, use the actual token count
        if n_tokens != n_vehicles * timesteps:
            # Fallback: treat all tokens as vehicles
            timesteps = 1
            n_vehicles = n_tokens
        
        # Sum attention across latents first
        attn_summed = attn.sum(axis=2)  # (n_samples, n_heads, n_vehicle_tokens)
        
        # Reshape and sum across timesteps to get per-vehicle attention
        if timesteps > 1:
            attn_reshaped = attn_summed.reshape(n_samples, n_heads, n_vehicles, timesteps)
            attn_per_vehicle = attn_reshaped.sum(axis=-1)  # (n_samples, n_heads, n_vehicles)
        else:
            attn_per_vehicle = attn_summed
        
        return attn_per_vehicle
    
    def aggregate_features(self) -> Dict[str, np.ndarray]:
        """
        Aggregate semantic features across all logs.
        
        Returns:
            Dictionary mapping feature names to arrays of shape (N_scenarios, N_vehicles).
        """
        if not self._logs:
            raise ValueError("No logs loaded. Call load_data() first.")
        
        # Check if semantic features are available
        has_features = any(bool(log.semantic_features) for log in self._logs)
        
        if not has_features:
            print("[Analyzer] Warning: No semantic features in logs. "
                  "HSI analysis requires training logs with semantic features.")
            return {}
        
        # Collect features from all logs
        feature_lists: Dict[str, List[np.ndarray]] = {f: [] for f in self.ANALYSIS_FEATURES}
        
        for log in self._logs:
            features = log.semantic_features
            if not features:
                continue
            
            for feature_name in self.ANALYSIS_FEATURES:
                if feature_name in features:
                    arr = features[feature_name]
                    # Ensure 2D: (n_samples, n_vehicles)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]
                    feature_lists[feature_name].append(arr)
        
        # Concatenate along scenario dimension
        for feature_name in self.ANALYSIS_FEATURES:
            if feature_lists[feature_name]:
                self._feature_arrays[feature_name] = np.concatenate(
                    feature_lists[feature_name], axis=0
                )
                print(f"[Analyzer] Feature '{feature_name}' shape: "
                      f"{self._feature_arrays[feature_name].shape}")
        
        return self._feature_arrays
    
    def compute_correlations(self) -> Dict[str, np.ndarray]:
        """
        Compute Pearson correlation between attention and each semantic feature.
        
        For each head h and feature f:
            ρ_h,f = corr(attention[h].flatten(), feature.flatten())
        
        Returns:
            Dictionary mapping feature names to correlation arrays of shape (n_heads,).
        """
        if self._attention_per_vehicle is None:
            self.aggregate_attention_by_vehicle()
        
        if not self._feature_arrays:
            self.aggregate_features()
        
        if not self._feature_arrays:
            return {}
        
        n_scenarios, n_heads, n_vehicles = self._attention_per_vehicle.shape
        correlations: Dict[str, np.ndarray] = {}
        
        for feature_name, feature_arr in self._feature_arrays.items():
            # Ensure shapes match
            if feature_arr.shape[0] != n_scenarios:
                print(f"[Analyzer] Warning: Shape mismatch for {feature_name}, skipping")
                continue
            
            # Compute correlation per head
            head_correlations = np.zeros(n_heads)
            
            for h in range(n_heads):
                attn_flat = self._attention_per_vehicle[:, h, :].flatten()
                feat_flat = feature_arr.flatten()
                
                # Handle NaN and Inf
                valid_mask = np.isfinite(attn_flat) & np.isfinite(feat_flat)
                
                if valid_mask.sum() < 10:
                    head_correlations[h] = 0.0
                    continue
                
                attn_valid = attn_flat[valid_mask]
                feat_valid = feat_flat[valid_mask]
                
                # Compute Pearson correlation
                r, _ = stats.pearsonr(attn_valid, feat_valid)
                head_correlations[h] = r if np.isfinite(r) else 0.0
            
            correlations[feature_name] = head_correlations
            print(f"[Analyzer] Correlations for '{feature_name}': {head_correlations}")
        
        return correlations
    
    def compute_hsi(self) -> HSIResult:
        """
        Compute Head Specialization Index for all heads.
        
        HSI_h = max_f |ρ_h,f|
        Primary_function_h = argmax_f |ρ_h,f|
        
        Returns:
            HSIResult object with all analysis results.
        """
        correlations = self.compute_correlations()
        
        if not correlations:
            raise ValueError("No correlations computed. Check if logs have semantic features.")
        
        # Get number of heads from attention shape
        n_heads = self._attention_per_vehicle.shape[1]
        
        # Build correlation matrix: (n_heads, n_features)
        feature_names = list(correlations.keys())
        corr_matrix = np.array([correlations[f] for f in feature_names]).T
        
        # Compute HSI (max absolute correlation per head)
        abs_corr = np.abs(corr_matrix)
        hsi_scores = abs_corr.max(axis=1)  # (n_heads,)
        primary_indices = abs_corr.argmax(axis=1)  # (n_heads,)
        
        # Get primary features and their correlations
        primary_features = [feature_names[i] for i in primary_indices]
        primary_correlations = np.array([
            corr_matrix[h, primary_indices[h]] for h in range(n_heads)
        ])
        
        # Assign functional labels
        head_labels = {}
        for h in range(n_heads):
            if hsi_scores[h] >= self.HSI_THRESHOLD:
                feat = primary_features[h]
                label_info = FEATURE_TO_LABEL.get(feat, DEFAULT_LABEL).copy()
                label_info['primary_feature'] = feat
                label_info['hsi'] = float(hsi_scores[h])
                label_info['correlation'] = float(primary_correlations[h])
                label_info['correlation_sign'] = 'negative' if primary_correlations[h] < 0 else 'positive'
            else:
                label_info = DEFAULT_LABEL.copy()
                label_info['primary_feature'] = None
                label_info['hsi'] = float(hsi_scores[h])
                label_info['correlation'] = 0.0
            
            head_labels[h] = label_info
        
        self._result = HSIResult(
            hsi_scores=hsi_scores,
            correlations=correlations,
            primary_features=primary_features,
            primary_correlations=primary_correlations,
            head_labels=head_labels
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("HEAD SPECIALIZATION ANALYSIS RESULTS")
        print("=" * 60)
        for h, label in head_labels.items():
            print(f"Head {h}: HSI={label['hsi']:.3f}, {label['name']}")
            if label.get('primary_feature'):
                print(f"         Primary: {label['primary_feature']} "
                      f"(ρ={label['correlation']:.3f})")
        print("=" * 60 + "\n")
        
        return self._result
    
    def export_registry(self, output_path: str):
        """
        Export head function registry to JSON file.
        
        Args:
            output_path: Path to output JSON file.
        """
        if self._result is None:
            self.compute_hsi()
        
        registry = {
            str(h): info for h, info in self._result.head_labels.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"[Analyzer] Exported registry to {output_path}")


# ==============================================================================
# Visualization
# ==============================================================================

class HeadVisualization:
    """Generate visualizations for head specialization analysis."""
    
    def __init__(self, analyzer: HeadSpecializationAnalyzer):
        """
        Initialize visualizer.
        
        Args:
            analyzer: HeadSpecializationAnalyzer with computed results.
        """
        self.analyzer = analyzer
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Plot heatmap of correlations between heads and features.
        
        Args:
            save_path: Optional path to save the figure.
        """
        result = self.analyzer.result
        if result is None:
            raise ValueError("No HSI results available. Call compute_hsi() first.")
        
        correlations = result.correlations
        feature_names = list(correlations.keys())
        n_heads = len(result.hsi_scores)
        
        # Build correlation matrix
        corr_matrix = np.array([correlations[f] for f in feature_names]).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"Head {h}" for h in range(n_heads)])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Pearson Correlation (ρ)')
        
        # Add text annotations
        for i in range(n_heads):
            for j in range(len(feature_names)):
                val = corr_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        
        ax.set_title('Head-Feature Correlation Heatmap')
        ax.set_xlabel('Semantic Feature')
        ax.set_ylabel('Attention Head')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved heatmap to {save_path}")
        
        plt.show()
        return fig
    
    def plot_hsi_bar(self, save_path: Optional[str] = None):
        """
        Plot bar chart of HSI scores per head.
        
        Args:
            save_path: Optional path to save the figure.
        """
        result = self.analyzer.result
        if result is None:
            raise ValueError("No HSI results available. Call compute_hsi() first.")
        
        n_heads = len(result.hsi_scores)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if hsi >= self.analyzer.HSI_THRESHOLD else '#95a5a6'
                  for hsi in result.hsi_scores]
        
        bars = ax.bar(range(n_heads), result.hsi_scores, color=colors, edgecolor='black')
        
        # Add threshold line
        ax.axhline(y=self.analyzer.HSI_THRESHOLD, color='red', linestyle='--',
                   label=f'Specialization Threshold ({self.analyzer.HSI_THRESHOLD})')
        
        # Add labels
        for i, (bar, label_info) in enumerate(zip(bars, result.head_labels.values())):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    label_info['name'].replace(' ', '\n'), ha='center', va='bottom',
                    fontsize=9)
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Head Specialization Index (HSI)')
        ax.set_title('Head Specialization Index per Attention Head')
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f'Head {h}' for h in range(n_heads)])
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved HSI bar chart to {save_path}")
        
        plt.show()
        return fig
    
    def plot_attention_vs_feature(
        self,
        head_idx: int,
        feature_name: str,
        save_path: Optional[str] = None
    ):
        """
        Scatter plot of attention vs. feature value for a specific head.
        
        Args:
            head_idx: Index of the attention head.
            feature_name: Name of the semantic feature.
            save_path: Optional path to save the figure.
        """
        attn = self.analyzer._attention_per_vehicle
        features = self.analyzer._feature_arrays
        
        if attn is None or feature_name not in features:
            raise ValueError("Attention or feature data not available.")
        
        attn_flat = attn[:, head_idx, :].flatten()
        feat_flat = features[feature_name].flatten()
        
        # Filter valid values
        valid = np.isfinite(attn_flat) & np.isfinite(feat_flat)
        attn_valid = attn_flat[valid]
        feat_valid = feat_flat[valid]
        
        # Subsample for plotting if too many points
        max_points = 5000
        if len(attn_valid) > max_points:
            indices = np.random.choice(len(attn_valid), max_points, replace=False)
            attn_valid = attn_valid[indices]
            feat_valid = feat_valid[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(feat_valid, attn_valid, alpha=0.3, s=10)
        
        # Add trend line
        z = np.polyfit(feat_valid, attn_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(feat_valid.min(), feat_valid.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend line')
        
        # Add correlation text
        r = self.analyzer.result.correlations[feature_name][head_idx]
        ax.text(0.05, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel(f'Head {head_idx} Attention')
        ax.set_title(f'Head {head_idx} Attention vs. {feature_name}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved scatter plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_all(self, output_dir: str):
        """
        Generate all visualizations and save to output directory.
        
        Args:
            output_dir: Directory to save all figures.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation heatmap
        self.plot_correlation_heatmap(os.path.join(output_dir, 'correlation_heatmap.png'))
        
        # HSI bar chart
        self.plot_hsi_bar(os.path.join(output_dir, 'hsi_scores.png'))
        
        # Scatter plots for specialized heads
        result = self.analyzer.result
        for h, label_info in result.head_labels.items():
            if label_info.get('primary_feature') and label_info['hsi'] >= self.analyzer.HSI_THRESHOLD:
                feat = label_info['primary_feature']
                self.plot_attention_vs_feature(
                    h, feat,
                    os.path.join(output_dir, f'scatter_head{h}_{feat}.png')
                )
        
        print(f"[Visualization] All figures saved to {output_dir}")


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """Command-line interface for head specialization analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Head Specialization Analysis for XAI'
    )
    parser.add_argument(
        '--feature_dir', '-f',
        type=str,
        required=True,
        help='Directory containing training logs (semantic features)'
    )
    parser.add_argument(
        '--attention_dir', '-a',
        type=str,
        required=True,
        help='Directory containing offline extractions (attention weights)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='hsi_results',
        help='Directory for output files (default: hsi_results)'
    )
    parser.add_argument(
        '--registry_path', '-r',
        type=str,
        default=None,
        help='Path for head function registry JSON (default: output_dir/head_functions.json)'
    )
    parser.add_argument(
        '--step_start',
        type=int,
        default=None,
        help='Start step for filtering logs'
    )
    parser.add_argument(
        '--step_end',
        type=int,
        default=None,
        help='End step for filtering logs'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set step range
    step_range = None
    if args.step_start is not None or args.step_end is not None:
        step_range = (
            args.step_start or 0,
            args.step_end or float('inf')
        )
    
    # Run analysis
    print(f"\n{'='*60}")
    print("HEAD SPECIALIZATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Feature directory: {args.feature_dir}")
    print(f"Attention directory: {args.attention_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load and analyze
    analyzer = HeadSpecializationAnalyzer(
        feature_dir=args.feature_dir,
        attention_dir=args.attention_dir
    )
    analyzer.load_data(step_range=step_range)
    analyzer.compute_hsi()
    
    # Export registry
    registry_path = args.registry_path or os.path.join(args.output_dir, 'head_functions.json')
    analyzer.export_registry(registry_path)
    
    # Generate visualizations
    viz = HeadVisualization(analyzer)
    viz.plot_all(args.output_dir)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Registry: {registry_path}")
    print(f"Visualizations: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
