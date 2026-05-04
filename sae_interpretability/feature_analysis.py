"""Grouped analysis and visualisation for SAE feature annotations.

Given the pre-computed annotation JSON (from ``feature_annotator.annotate``)
and the raw data (features matrix + telemetry dict), this module generates:

1. **Best Feature Per Concept** — horizontal bar chart of |ρ| per telemetry
   field, showing the single strongest feature for each concept.
2. **Label Distribution** — bar chart of how many features received each
   label, plus a pie chart of dead vs active features.
3. **Concept Group Heatmap** — top-3 features per label group, full ρ
   profile across all telemetry fields.
4. **Multi-Variate Clustermap** — ALL active features × telemetry fields,
   hierarchically clustered with dendrogram.
5. **Selectivity Histogram** — distribution of activation counts across
   all features.

All figures are saved to ``data/sae_interpretability/figures/``.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — safe for servers / SSH sessions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Colour palette & style defaults
# ---------------------------------------------------------------------------

_PALETTE = "viridis"
_DIVERGING_CMAP = "RdBu_r"  # red = positive ρ, blue = negative
_FIG_DIR = os.path.join("data", "sae_interpretability", "figures")

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.15,
    rc={
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    },
)


def _ensure_fig_dir(fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)


def _strip_rho_suffix(label: str) -> str:
    """Remove the trailing (ρ=…) or (z=…) suffix for cleaner grouping.

    'high_ego_speed (ρ=+0.812)' → 'high_ego_speed'
    """
    return re.sub(r"\s*\(.*\)\s*$", "", label).strip()


# ===================================================================
# 1. Best Feature Per Concept
# ===================================================================

def best_feature_per_concept(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    fig_dir: str = _FIG_DIR,
) -> None:
    """For each telemetry field, find the single feature with highest |ρ|.

    Prints the table and saves a horizontal bar chart.
    """
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]

    # Build best-per-concept lookup
    best: Dict[str, Tuple[int, float]] = {}  # concept → (feat_idx, rho)

    for ann in active:
        rho_scores = ann.get("all_rho_scores", {})
        for field, rho in rho_scores.items():
            abs_rho = abs(rho)
            if field not in best or abs_rho > abs(best[field][1]):
                best[field] = (ann["feature_idx"], rho)

    # Print table
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║          Best Feature Per Telemetry Concept                 ║")
    print("╠══════════════════════════════════╦══════════╦═══════════════╣")
    print("║ Concept                          ║ Feature  ║     |ρ|      ║")
    print("╠══════════════════════════════════╬══════════╬═══════════════╣")
    for concept in sorted(best.keys()):
        feat_idx, rho = best[concept]
        print(f"║ {concept:<32s} ║ {feat_idx:>8d} ║ {abs(rho):>12.4f} ║")
    print("╚══════════════════════════════════╩══════════╩═══════════════╝")

    # Horizontal bar chart
    concepts = sorted(best.keys(), key=lambda c: abs(best[c][1]))
    rho_vals = [abs(best[c][1]) for c in concepts]
    colours = sns.color_palette("mako", n_colors=len(concepts))

    fig, ax = plt.subplots(figsize=(8, max(4, len(concepts) * 0.45)))
    bars = ax.barh(concepts, rho_vals, color=colours, edgecolor="white", linewidth=0.5)

    # Annotate feature indices on bars
    for bar, concept in zip(bars, concepts):
        feat_idx, rho = best[concept]
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"f{feat_idx} (ρ={rho:+.3f})",
            va="center", fontsize=8, color="#333",
        )

    ax.set_xlabel("|Spearman ρ|")
    ax.set_title("Best SAE Feature Per Telemetry Concept")
    ax.set_xlim(0, max(rho_vals) * 1.25 if rho_vals else 1.0)
    fig.tight_layout()
    path = os.path.join(fig_dir, "best_per_concept.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {path}")


# ===================================================================
# 2. Label Distribution
# ===================================================================

def label_distribution(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Bar chart of label counts + pie chart of dead vs active."""
    _ensure_fig_dir(fig_dir)

    counts: Dict[str, int] = defaultdict(int)
    n_dead = 0
    n_active = 0

    for ann in annotations:
        raw_label = ann.get("label", "unknown")
        if raw_label == "dead":
            n_dead += 1
        else:
            n_active += 1
            clean = _strip_rho_suffix(raw_label)
            counts[clean] += 1

    # Sort by count descending
    sorted_labels = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    sorted_counts = [counts[k] for k in sorted_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(sorted_labels) * 0.35)),
                             gridspec_kw={"width_ratios": [2.5, 1]})

    # --- Bar chart ---
    palette = sns.color_palette("mako_r", n_colors=len(sorted_labels))
    ax_bar = axes[0]
    ax_bar.barh(sorted_labels[::-1], sorted_counts[::-1],
                color=palette[::-1], edgecolor="white", linewidth=0.5)
    ax_bar.set_xlabel("Number of Features")
    ax_bar.set_title("Feature Label Distribution")
    for i, (lbl, cnt) in enumerate(zip(sorted_labels[::-1], sorted_counts[::-1])):
        ax_bar.text(cnt + 0.3, i, str(cnt), va="center", fontsize=9, color="#333")

    # --- Pie chart ---
    ax_pie = axes[1]
    pie_data = [n_active, n_dead]
    pie_labels = [f"Active\n({n_active})", f"Dead\n({n_dead})"]
    pie_colours = ["#4CAF50", "#9E9E9E"]
    wedges, texts, autotexts = ax_pie.pie(
        pie_data, labels=pie_labels, colors=pie_colours,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("white")
        t.set_fontweight("bold")
    ax_pie.set_title("Dead vs Active Features")

    fig.tight_layout()
    path = os.path.join(fig_dir, "label_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {path}")

    # Print summary
    print("\n--- Label Distribution ---")
    for lbl, cnt in zip(sorted_labels, sorted_counts):
        print(f"  {lbl:<40s}  {cnt:>4d} features")
    print(f"  {'DEAD':<40s}  {n_dead:>4d} features")
    print(f"  Total: {n_active + n_dead} features ({n_active} active, {n_dead} dead)")


# ===================================================================
# 3. Top Features Per Concept Group — Heatmap
# ===================================================================

def concept_group_heatmap(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    top_n: int = 3,
    fig_dir: str = _FIG_DIR,
) -> None:
    """Group features by label, take top-N per group, plot ρ heatmap."""
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]

    # Group by cleaned label
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for ann in active:
        clean = _strip_rho_suffix(ann.get("label", ""))
        groups[clean].append(ann)

    # For each group, keep top_n by |ρ| (selectivity_score)
    rows_labels: List[str] = []
    rows_data: List[List[float]] = []

    for group_label in sorted(groups.keys()):
        members = sorted(groups[group_label],
                         key=lambda a: a.get("selectivity_score", 0),
                         reverse=True)[:top_n]
        for ann in members:
            rho_scores = ann.get("all_rho_scores", {})
            row = [rho_scores.get(k, 0.0) for k in tel_keys]
            rows_data.append(row)
            rows_labels.append(f"f{ann['feature_idx']} ({group_label})")

    if not rows_data:
        print("[Analysis] No active features for concept group heatmap.")
        return

    mat = np.array(rows_data)

    fig_height = max(5, len(rows_labels) * 0.35)
    fig_width = max(8, len(tel_keys) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        mat,
        xticklabels=tel_keys,
        yticklabels=rows_labels,
        cmap=_DIVERGING_CMAP,
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        linecolor="#eee",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.75},
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        ax=ax,
    )
    ax.set_title(f"Top-{top_n} Features Per Concept Group — Full ρ Profile")
    ax.set_xlabel("Telemetry Field")
    ax.set_ylabel("Feature (Group)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, "concept_group_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {path}")


# ===================================================================
# 4. Multi-Variate Clustermap
# ===================================================================

def feature_telemetry_clustermap(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Clustered heatmap of ALL active features × telemetry fields.

    Uses ``sns.clustermap`` with a dendrogram to reveal groups of features
    encoding similar compound concepts.
    """
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]

    if not active:
        print("[Analysis] No active features for clustermap.")
        return

    # Build the correlation matrix: [n_active, n_tel]
    feat_indices: List[int] = []
    rows: List[List[float]] = []
    for ann in active:
        rho_scores = ann.get("all_rho_scores", {})
        rows.append([rho_scores.get(k, 0.0) for k in tel_keys])
        feat_indices.append(ann["feature_idx"])

    mat = np.array(rows)

    # For very large matrices, cap at 200 features to keep the figure readable
    MAX_FEATURES = 200
    if mat.shape[0] > MAX_FEATURES:
        # Keep the most selective features
        selectivity = np.max(np.abs(mat), axis=1)
        top_idx = np.argsort(selectivity)[-MAX_FEATURES:]
        mat = mat[top_idx]
        feat_indices = [feat_indices[i] for i in top_idx]
        print(f"[Analysis] Clustermap: capped to top {MAX_FEATURES} most selective features.")

    import pandas as pd
    df = pd.DataFrame(mat, columns=tel_keys,
                      index=[f"f{i}" for i in feat_indices])

    try:
        g = sns.clustermap(
            df,
            cmap=_DIVERGING_CMAP,
            center=0,
            vmin=-1, vmax=1,
            figsize=(max(10, len(tel_keys) * 0.8),
                     max(8, min(len(feat_indices) * 0.12, 30))),
            linewidths=0,
            cbar_kws={"label": "Spearman ρ"},
            dendrogram_ratio=(0.12, 0.08),
            method="ward",
            metric="euclidean",
            yticklabels=True if len(feat_indices) <= 80 else False,
            xticklabels=True,
        )
        g.ax_heatmap.set_xlabel("Telemetry Field")
        g.ax_heatmap.set_ylabel("SAE Feature")
        g.fig.suptitle(
            "Multi-Variate Feature Profiles — Clustered Heatmap",
            y=1.02, fontsize=14, fontweight="bold",
        )
        path = os.path.join(fig_dir, "feature_telemetry_clustermap.png")
        g.savefig(path, bbox_inches="tight")
        plt.close(g.fig)
        print(f"[Analysis] Saved → {path}")
    except Exception as e:
        print(f"[Analysis] Clustermap failed (likely too few features): {e}")


# ===================================================================
# 5. Selectivity Histogram
# ===================================================================

def selectivity_histogram(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Histogram of activation counts across all features."""
    _ensure_fig_dir(fig_dir)

    n_acts = [a.get("n_activations", 0) for a in annotations]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use log-spaced bins for better visualisation of long-tail distributions
    n_acts_arr = np.array(n_acts)
    nonzero = n_acts_arr[n_acts_arr > 0]

    if len(nonzero) > 0:
        bins = np.logspace(
            np.log10(max(1, nonzero.min())),
            np.log10(nonzero.max() + 1),
            50,
        )
        ax.hist(nonzero, bins=bins, color="#5C6BC0", edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.set_xscale("log")
    else:
        ax.hist(n_acts, bins=30, color="#5C6BC0", edgecolor="white",
                linewidth=0.5, alpha=0.85)

    # Mark dead threshold
    dead_threshold_line = ax.axvline(
        x=10, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.legend([dead_threshold_line], ["Dead threshold (n < 10)"],
              loc="upper right", fontsize=9)

    ax.set_xlabel("Number of Timesteps Feature Fires On (log scale)")
    ax.set_ylabel("Count of Features")
    ax.set_title("Feature Selectivity — Activation Count Distribution")

    # Add summary stats as text box
    n_dead = sum(1 for n in n_acts if n < 10)
    median_act = int(np.median(n_acts)) if n_acts else 0
    mean_act = np.mean(n_acts) if n_acts else 0
    textstr = (f"Total: {len(n_acts)}\n"
               f"Dead (n<10): {n_dead}\n"
               f"Median n: {median_act}\n"
               f"Mean n: {mean_act:.0f}")
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="#ccc", alpha=0.9)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    fig.tight_layout()
    path = os.path.join(fig_dir, "selectivity_histogram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {path}")


# ===================================================================
# Orchestrator
# ===================================================================

def run_analysis(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Run all grouped analyses and generate all figures.

    Args:
        annotations: The list of per-feature annotation dicts (from the
            ``feature_annotator.annotate`` return value, or loaded from
            the annotations JSON under the ``'annotations'`` key).
        tel_keys: Ordered list of telemetry field names (column names for
            the correlation matrices).
        fig_dir: Output directory for figures.
    """
    print("\n" + "=" * 64)
    print("  SAE Feature Analysis — Grouped Visualisations")
    print("=" * 64)

    best_feature_per_concept(annotations, tel_keys, fig_dir)
    label_distribution(annotations, fig_dir)
    concept_group_heatmap(annotations, tel_keys, fig_dir=fig_dir)
    feature_telemetry_clustermap(annotations, tel_keys, fig_dir=fig_dir)
    selectivity_histogram(annotations, fig_dir)

    print("\n" + "=" * 64)
    print(f"  All figures saved to {fig_dir}/")
    print("=" * 64 + "\n")
