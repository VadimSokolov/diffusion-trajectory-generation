#!/usr/bin/env python3
"""
Evaluation script for CSDI-generated trajectories.

Compares synthetic trajectories against real micro-trips using comprehensive metrics.
"""

import sys
import os
import glob
import pickle
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for importing evaluation_metrics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdv'))

from evaluation_metrics import (
    run_full_evaluation,
    create_evaluation_report,
    compute_safd,
    safd_wasserstein_distance,
    boundary_violation_rate,
    speed_distribution_metrics,
    acceleration_distribution_metrics,
)


def load_real_trajectories(data_path: str, max_files: int = None) -> list:
    """Load real micro-trip trajectories from CSV files."""
    files = glob.glob(os.path.join(data_path, "**", "results_trip_*.csv"), recursive=True)
    
    if max_files:
        files = files[:max_files]
    
    print(f"Loading {len(files)} real trajectory files...")
    
    trajectories = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "speedMps" in df.columns:
                speed = df["speedMps"].values.astype(np.float32)
                if len(speed) >= 50:  # Min duration filter
                    trajectories.append(speed)
        except Exception as e:
            continue
    
    print(f"Loaded {len(trajectories)} valid real trajectories")
    return trajectories


def load_csdi_trajectories(csv_path: str = None, pkl_path: str = None) -> list:
    """Load CSDI-generated trajectories from CSV or pickle."""
    
    if pkl_path and os.path.exists(pkl_path):
        print(f"Loading from pickle: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data['trajectories']
    
    if csv_path and os.path.exists(csv_path):
        print(f"Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Get length column if available
        if 'length' in df.columns:
            lengths = df['length'].values
            # Remove metadata columns
            data_cols = [c for c in df.columns if c not in ['length', 'avg_speed_target']]
            data = df[data_cols].values
            
            trajectories = []
            for i, length in enumerate(lengths):
                traj = data[i, :int(length)]
                trajectories.append(traj)
            return trajectories
        else:
            # Assume all columns are trajectory data
            return [row[~np.isnan(row)] for _, row in df.iterrows()]
    
    raise FileNotFoundError(f"No valid trajectory file found: csv={csv_path}, pkl={pkl_path}")


def plot_comparison(real_trips, synth_trips, output_path: str):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Speed distribution
    ax = axes[0, 0]
    real_speeds = np.concatenate(real_trips)
    synth_speeds = np.concatenate(synth_trips)
    ax.hist(real_speeds, bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(synth_speeds, bins=50, alpha=0.5, label='CSDI', density=True)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution')
    ax.legend()
    
    # 2. Acceleration distribution
    ax = axes[0, 1]
    real_accels = np.concatenate([np.diff(t) for t in real_trips])
    synth_accels = np.concatenate([np.diff(t) for t in synth_trips])
    ax.hist(real_accels, bins=50, alpha=0.5, label='Real', density=True, range=(-3, 3))
    ax.hist(synth_accels, bins=50, alpha=0.5, label='CSDI', density=True, range=(-3, 3))
    ax.set_xlabel('Acceleration (m/s²)')
    ax.set_ylabel('Density')
    ax.set_title('Acceleration Distribution')
    ax.legend()
    
    # 3. Duration distribution
    ax = axes[0, 2]
    real_durs = [len(t) for t in real_trips]
    synth_durs = [len(t) for t in synth_trips]
    ax.hist(real_durs, bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(synth_durs, bins=30, alpha=0.5, label='CSDI', density=True)
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Density')
    ax.set_title('Duration Distribution')
    ax.legend()
    
    # 4. SAFD Real
    ax = axes[1, 0]
    safd_real, speed_bins, accel_bins = compute_safd(real_trips)
    im = ax.imshow(safd_real.T, origin='lower', aspect='auto',
                   extent=[speed_bins[0], speed_bins[-1], accel_bins[0], accel_bins[-1]],
                   cmap='Blues')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('SAFD - Real')
    plt.colorbar(im, ax=ax)
    
    # 5. SAFD Synthetic
    ax = axes[1, 1]
    safd_synth, _, _ = compute_safd(synth_trips, speed_bins, accel_bins)
    im = ax.imshow(safd_synth.T, origin='lower', aspect='auto',
                   extent=[speed_bins[0], speed_bins[-1], accel_bins[0], accel_bins[-1]],
                   cmap='Oranges')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('SAFD - CSDI')
    plt.colorbar(im, ax=ax)
    
    # 6. Sample trajectories
    ax = axes[1, 2]
    for i, t in enumerate(synth_trips[:5]):
        ax.plot(t, alpha=0.7, label=f'Synth {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Sample CSDI Trajectories')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSDI trajectories')
    parser.add_argument('--real_data', type=str, default='../../data/Microtrips',
                        help='Path to real micro-trips')
    parser.add_argument('--synth_csv', type=str, default=None,
                        help='Path to CSDI synthetic CSV')
    parser.add_argument('--synth_pkl', type=str, default=None,
                        help='Path to CSDI synthetic pickle')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--max_real', type=int, default=None,
                        help='Max number of real trajectories to load')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("CSDI Trajectory Evaluation")
    print("=" * 60)
    
    # Auto-detect synthetic file if not specified
    if args.synth_csv is None and args.synth_pkl is None:
        # Look for most recent CSV
        csv_files = glob.glob(os.path.join(args.output_dir, "csdi_synthetic_*.csv"))
        if csv_files:
            args.synth_csv = max(csv_files, key=os.path.getctime)
            print(f"Auto-detected synthetic file: {args.synth_csv}")
        else:
            print("ERROR: No synthetic file found. Specify --synth_csv or --synth_pkl")
            return
    
    # Load data
    real_trips = load_real_trajectories(args.real_data, args.max_real)
    synth_trips = load_csdi_trajectories(args.synth_csv, args.synth_pkl)
    
    print(f"\nReal trajectories: {len(real_trips)}")
    print(f"Synthetic trajectories: {len(synth_trips)}")
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Running Evaluation Metrics")
    print("=" * 60)
    
    metrics = run_full_evaluation(real_trips, synth_trips, verbose=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    key_metrics = [
        ("SAFD Wasserstein", "safd_wasserstein"),
        ("Speed Wasserstein", "speed_wasserstein"),
        ("Accel Wasserstein", "accel_wasserstein"),
        ("MMD", "mmd"),
        ("Discriminative Score", "discriminative_score"),
        ("Boundary Violation Rate", "boundary_any_violation_rate"),
        ("TSTR MAE", "tstr_mae"),
    ]
    
    print("\nKey Metrics:")
    print("-" * 40)
    for name, key in key_metrics:
        if key in metrics:
            print(f"  {name}: {metrics[key]:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"csdi_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types
        json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}
        json.dump(json_metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    
    # Create report DataFrame
    report = create_evaluation_report(metrics, "CSDI")
    report_path = os.path.join(args.output_dir, f"csdi_report_{timestamp}.csv")
    report.to_csv(report_path, index=False)
    print(f"Saved report: {report_path}")
    
    # Create comparison plots
    plot_path = os.path.join(args.output_dir, f"csdi_evaluation_{timestamp}.png")
    plot_comparison(real_trips, synth_trips, plot_path)
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Real':<15} {'CSDI':<15}")
    print("-" * 60)
    print(f"{'Mean Speed (m/s)':<30} {np.mean([np.mean(t) for t in real_trips]):<15.2f} {np.mean([np.mean(t) for t in synth_trips]):<15.2f}")
    print(f"{'Std Speed (m/s)':<30} {np.mean([np.std(t) for t in real_trips]):<15.2f} {np.mean([np.std(t) for t in synth_trips]):<15.2f}")
    print(f"{'Mean Duration (s)':<30} {np.mean([len(t) for t in real_trips]):<15.1f} {np.mean([len(t) for t in synth_trips]):<15.1f}")
    print(f"{'Max Speed (m/s)':<30} {np.max([np.max(t) for t in real_trips]):<15.2f} {np.max([np.max(t) for t in synth_trips]):<15.2f}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

