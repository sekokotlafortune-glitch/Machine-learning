"""
 Distributed ML Demo
Visualization & Results Dashboard
===================================
Generates publication-quality charts for:
  1. Loss & Accuracy curves
  2. Confusion Matrix
  3. Per-class performance (Precision/Recall/F1)
  4. Training time comparison (Single vs Distributed)
  5. Data Partitioning across nodes
  6. Comprehensive Summary Dashboard

Usage:
    python data_partitioning_viz.py
"""

import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# ── Style configuration ──
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# CONFIGURATION & PATHS
# ──────────────────────────────────────────────
RESULTS_DIR = "."
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

C_TRAIN   = "#1565C0"   # deep blue
C_VAL     = "#E53935"   # red
C_SINGLE  = "#1565C0"   # blue
C_DIST    = "#2E7D32"   # green
BG        = "#FAFAFA"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def save(fig, name, tight=True):
    path = os.path.join(PLOTS_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅  Saved: {path}")

def style_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=9, framealpha=0.7)
    ax.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════
# DATA PARTITIONING VISUALIZATION (Core feature)
# ══════════════════════════════════════════════════════════════════
def plot_data_partitioning(num_workers=3):
    print("  [1/6]  Visualizing data partitioning …")
    transform = transforms.Compose([transforms.ToTensor()])
    
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    except Exception as e:
        print(f"  ⚠️  Could not load CIFAR-10 for partitioning viz: {e}")
        return

    samples_per_worker = []
    for rank in range(num_workers):
        sampler = DistributedSampler(trainset, num_replicas=num_workers, rank=rank, shuffle=False)
        samples_per_worker.append(len(list(sampler)))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    workers = [f"Node {i}" for i in range(num_workers)]
    bars = ax.bar(workers, samples_per_worker, color=C_DIST, edgecolor="white", alpha=0.8)
    
    for bar, v in zip(bars, samples_per_worker):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{v}", ha='center', fontweight='bold', color=C_DIST)

    style_ax(ax, f"Data Partitioning Across {num_workers} Nodes", "Computing Nodes", "Samples Assigned", legend=False)
    ax.set_ylim(0, max(samples_per_worker) * 1.2)
    save(fig, "01_data_partitioning.png")

# ══════════════════════════════════════════════════════════════════
# TIME COMPARISON
# ══════════════════════════════════════════════════════════════════
def plot_time_comparison(single_time, dist_time, num_workers):
    print("  [2/6]  Comparing training times …")
    speedup = single_time / dist_time
    efficiency = (speedup / num_workers) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    
    # Raw Time
    ax = axes[0]
    labels = ["Single", f"Distributed\n({num_workers} nodes)"]
    times = [single_time, dist_time]
    bars = ax.bar(labels, times, color=[C_SINGLE, C_DIST], width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{t:.1f}s", ha='center', fontweight='bold')
    style_ax(ax, "Training Time Comparison", "", "Seconds", legend=False)

    # Speedup
    ax = axes[1]
    ax.bar(["Actual Speedup", "Ideal Speedup"], [speedup, num_workers], color=[C_DIST, "#BBDEFB"], width=0.5)
    ax.text(0, speedup + 0.1, f"{speedup:.2f}x", ha='center', fontweight='bold')
    style_ax(ax, f"Speedup Efficiency: {efficiency:.1f}%", "", "Speedup Factor", legend=False)
    
    save(fig, "02_time_comparison.png")

# ══════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════
def plot_summary_dashboard(single_stats, dist_stats):
    print("  [3/6]  Generating summary dashboard …")
    num_workers = dist_stats['workers']
    single_time = single_stats['total_time']
    dist_time = dist_stats['total_time']
    speedup = single_time / dist_time
    efficiency = (speedup / num_workers) * 100

    fig = plt.figure(figsize=(16, 9), facecolor="#F0F4F8")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.text(0.5, 0.96, "Distributed ML Performance Dashboard", ha="center", fontsize=20, fontweight="bold", color="#1A237E")

    # Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["Single", "Distributed"], [single_time, dist_time], color=[C_SINGLE, C_DIST])
    ax1.set_title("Training Time (s)", fontweight="bold")

    # Speedup
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(["Actual", "Ideal"], [speedup, num_workers], color=[C_DIST, "#BBDEFB"])
    ax2.set_title("Speedup Factor", fontweight="bold")

    # Metrics Panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics = [
        ("Workers", f"{num_workers} Nodes"),
        ("Speedup", f"{speedup:.2f}x"),
        ("Efficiency", f"{efficiency:.1f}%"),
        ("Time Saved", f"{single_time - dist_time:.1f}s")
    ]
    for i, (k, v) in enumerate(metrics):
        ax3.text(0.1, 0.8 - i*0.2, f"{k}:", fontsize=14, fontweight="bold")
        ax3.text(0.6, 0.8 - i*0.2, v, fontsize=14, color=C_DIST if i>0 else "black")

    # Note: Accuracy and Loss curves would go here if history was available in the JSON
    # For now, we show the final loss
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    ax4.text(0.5, 0.5, f"Final Distributed Loss: {dist_stats['final_loss']:.4f}\n"
                     f"Training completed on: {dist_stats.get('timestamp', 'N/A')}", 
             ha="center", fontsize=16, style='italic')

    save(fig, "00_summary_dashboard.png", tight=False)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*55)
    print("  Distributed ML Visualization Dashboard")
    print("="*55)

    single_res = load_json("results_local_1.json")
    dist_res = load_json("results_distributed_3.json")

    if not single_res or not dist_res:
        print("  ⚠️  Missing results files! Please run training first.")
        print("  Looking for: results_local_1.json and results_distributed_3.json")
        return

    # 1. Partitioning
    plot_data_partitioning(num_workers=dist_res['workers'])

    # 2. Time Comparison
    plot_time_comparison(single_res['total_time'], dist_res['total_time'], dist_res['workers'])

    # 3. Dashboard
    plot_summary_dashboard(single_res, dist_res)

    print(f"\n{'='*55}")
    print(f"  ✅  Charts generated and saved to: {os.path.abspath(PLOTS_DIR)}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
