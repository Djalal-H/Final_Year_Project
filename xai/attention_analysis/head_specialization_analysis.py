"""Head Specialization Analysis for XAI.

This module analyzes attention head behavior to discover functional specialization
using extracted attention weights and semantic features from offline_extraction.py.

Key Concepts:
- HSI (Head Specialization Index): Measures how strongly a head correlates with 
  specific semantic features (TTC, distance, etc.)
- Primary Function: The semantic feature with strongest correlation for each head
- Head Function Registry: Maps heads to interpretable function labels

Usage:
    # Analyze a single checkpoint extraction
    analyzer = HeadSpecializationAnalyzer(
        extraction_dir="./extractions"
    )
    analyzer.load_data()
    results = analyzer.compute_hsi()
    analyzer.export_registry("head_functions.json")
    analyzer.visualize_all("./hsi_plots")
    
    # Or via CLI:
    python head_specialization_analysis.py \\
        --extraction_dir ./extractions \\
        --output_dir ./hsi_results
"""

import argparse
import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class HSIResult:
    """Results from Head Specialization Index computation."""
    hsi_scores: np.ndarray  # Shape: (n_heads,)
    correlations: Dict[str, np.ndarray]  # Feature -> (n_heads,) correlations
    primary_features: List[str]  # Primary feature per head
    primary_correlations: np.ndarray  # Correlation value for primary feature
    head_labels: Dict[int, Dict[str, Any]]  # Head index -> label info
    checkpoint_step: int  # Training step of the checkpoint


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
# Head Specialization Analyzer
# ==============================================================================

class HeadSpecializationAnalyzer:
    """
    Analyze attention head specialization through correlation analysis.
    
    This class loads extraction outputs from offline_extraction.py and computes
    the Head Specialization Index (HSI) for each attention head.
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
    
    def __init__(self, extraction_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            extraction_dir: Directory containing extraction output files from offline_extraction.py.
        """
        self.extraction_dir = extraction_dir
        self._extraction_data = None
        self._result: Optional[HSIResult] = None
        
        # Aggregated data for analysis
        self._attention_per_vehicle: Optional[np.ndarray] = None  # (N, H, V)
        self._feature_arrays: Dict[str, np.ndarray] = {}  # Feature -> (N, V)
        
    def load_data(self, extraction_file: Optional[str] = None):
        """Load extraction data from pickle file.
        
        Args:
            extraction_file: Specific extraction file to load. If None, loads the first .pkl file found.
        """
        if extraction_file is None:
            # Find extraction files
            pattern = os.path.join(self.extraction_dir, "extraction_*.pkl")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No extraction files found in {self.extraction_dir}")
            extraction_file = files[0]
            print(f"[HSI Analyzer] Auto-selected: {os.path.basename(extraction_file)}")
        else:
            extraction_file = os.path.join(self.extraction_dir, extraction_file)
        
        print(f"[HSI Analyzer] Loading extraction from {extraction_file}")
        
        with open(extraction_file, 'rb') as f:
            self._extraction_data = pickle.load(f)
        
        print(f"[HSI Analyzer] Loaded {self._extraction_data['n_scenarios']} scenarios")
        print(f"[HSI Analyzer] Checkpoint: {self._extraction_data['checkpoint']}")
        print(f"[HSI Analyzer] Training step: {self._extraction_data['step']}")
        
        return self
    
    def aggregate_data(self):
        """Aggregate attention and features across all scenarios."""
        if self._extraction_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        scenarios = self._extraction_data['scenarios']
        
        # Collect attention per vehicle from all scenarios
        attention_list = []
        for scenario in scenarios:
            attn = scenario.get('attention_per_vehicle')
            if attn is not None:
                # Shape should be (n_heads, n_vehicles)
                attention_list.append(attn)
        
        if not attention_list:
            raise ValueError("No valid attention data found in scenarios")
        
        # Stack to (n_scenarios, n_heads, n_vehicles)
        self._attention_per_vehicle = np.stack(attention_list, axis=0)
        print(f"[HSI Analyzer] Attention shape: {self._attention_per_vehicle.shape}")
        
        # Collect semantic features
        for feature_name in self.ANALYSIS_FEATURES:
            feature_list = []
            for scenario in scenarios:
                features = scenario.get('semantic_features', {})
                if feature_name in features:
                    feat = features[feature_name]
                    # Ensure 1D: (n_vehicles,)
                    if feat.ndim == 0:
                        feat = np.array([feat])
                    feature_list.append(feat)
            
            if feature_list:
                # Stack to (n_scenarios, n_vehicles)
                self._feature_arrays[feature_name] = np.stack(feature_list, axis=0)
                print(f"[HSI Analyzer] Feature '{feature_name}' shape: {self._feature_arrays[feature_name].shape}")
        
        return self
    
    def compute_correlations(self) -> Dict[str, np.ndarray]:
        """
        Compute Pearson correlation between attention and each semantic feature.
        
        For each head h and feature f:
            ρ_h,f = corr(attention[h].flatten(), feature.flatten())
        
        Returns:
            Dictionary mapping feature names to correlation arrays of shape (n_heads,).
        """
        if self._attention_per_vehicle is None:
            self.aggregate_data()
        
        if not self._feature_arrays:
            raise ValueError("No semantic features available for correlation")
        
        n_scenarios, n_heads, n_vehicles = self._attention_per_vehicle.shape
        correlations: Dict[str, np.ndarray] = {}
        
        for feature_name, feature_arr in self._feature_arrays.items():
            # Ensure shapes match
            if feature_arr.shape[0] != n_scenarios:
                print(f"[HSI Analyzer] Warning: Shape mismatch for {feature_name}, skipping")
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
                
                # Check for constant arrays (no variance)
                if np.std(attn_valid) < 1e-10 or np.std(feat_valid) < 1e-10:
                    # One or both arrays are constant → correlation undefined
                    head_correlations[h] = 0.0
                    continue
                
                # Compute Pearson correlation
                r, _ = stats.pearsonr(attn_valid, feat_valid)
                head_correlations[h] = r if np.isfinite(r) else 0.0
            
            correlations[feature_name] = head_correlations
            print(f"[HSI Analyzer] Correlations for '{feature_name}': {head_correlations}")
        
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
            raise ValueError("No correlations computed")
        
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
            head_labels=head_labels,
            checkpoint_step=self._extraction_data['step']
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("HEAD SPECIALIZATION ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Checkpoint Step: {self._result.checkpoint_step}")
        print(f"Number of Heads: {n_heads}")
        print("-" * 60)
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
            'checkpoint_step': self._result.checkpoint_step,
            'checkpoint': self._extraction_data['checkpoint'],
            'hsi_threshold': self.HSI_THRESHOLD,
            'heads': {
                str(h): info for h, info in self._result.head_labels.items()
            }
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"[HSI Analyzer] Exported registry to {output_path}")
    
    def visualize_all(self, output_dir: str):
        """Generate all visualizations.
        
        Args:
            output_dir: Directory to save plots.
        """
        if self._result is None:
            self.compute_hsi()
        
        os.makedirs(output_dir, exist_ok=True)
        
        viz = HeadVisualization(self)
        
        # Correlation heatmap
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        viz.plot_correlation_heatmap(save_path=heatmap_path)
        plt.close()
        
        # HSI bar chart
        hsi_bar_path = os.path.join(output_dir, "hsi_bar_chart.png")
        viz.plot_hsi_bar(save_path=hsi_bar_path)
        plt.close()
        
        # Scatter plots for specialized heads
        for h, label in self._result.head_labels.items():
            if label.get('primary_feature'):
                scatter_path = os.path.join(output_dir, f"head_{h}_{label['primary_feature']}.png")
                viz.plot_attention_vs_feature(
                    head_idx=h,
                    feature_name=label['primary_feature'],
                    save_path=scatter_path
                )
                plt.close()
        
        print(f"[HSI Analyzer] Saved visualizations to {output_dir}")


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
        result = self.analyzer._result
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
        
        ax.set_title(f'Head-Feature Correlation Heatmap (Step {result.checkpoint_step})')
        ax.set_xlabel('Semantic Feature')
        ax.set_ylabel('Attention Head')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved heatmap to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_hsi_bar(self, save_path: Optional[str] = None):
        """
        Plot bar chart of HSI scores per head.
        
        Args:
            save_path: Optional path to save the figure.
        """
        result = self.analyzer._result
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
        ax.set_title(f'Head Specialization Index (Step {result.checkpoint_step})')
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f'Head {h}' for h in range(n_heads)])
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualization] Saved HSI bar chart to {save_path}")
        else:
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
        r = self.analyzer._result.correlations[feature_name][head_idx]
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
        else:
            plt.show()
        
        return fig


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Head Specialization Analysis for attention mechanisms"
    )
    parser.add_argument(
        "--extraction_dir",
        type=str,
        required=True,
        help="Directory containing extraction output from offline_extraction.py"
    )
    parser.add_argument(
        "--extraction_file",
        type=str,
        default=None,
        help="Specific extraction file to analyze (default: auto-select first)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hsi_results",
        help="Directory to save analysis results (default: ./hsi_results)"
    )
    parser.add_argument(
        "--hsi_threshold",
        type=float,
        default=0.3,
        help="HSI threshold for specialization (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HeadSpecializationAnalyzer(extraction_dir=args.extraction_dir)
    analyzer.HSI_THRESHOLD = args.hsi_threshold
    
    # Run analysis
    print("=" * 60)
    print("HEAD SPECIALIZATION ANALYSIS")
    print("=" * 60)
    
    analyzer.load_data(extraction_file=args.extraction_file)
    results = analyzer.compute_hsi()
    
    # Export results
    os.makedirs(args.output_dir, exist_ok=True)
    
    registry_path = os.path.join(args.output_dir, "head_registry.json")
    analyzer.export_registry(registry_path)
    
    # Generate visualizations
    viz_dir = os.path.join(args.output_dir, "visualizations")
    analyzer.visualize_all(viz_dir)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
