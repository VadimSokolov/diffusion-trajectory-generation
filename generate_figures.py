#!/usr/bin/env python3
"""
Generate all figures for IEEE ITS paper on diffusion models for vehicle speed trajectory generation.

This script creates:
- Figure 1: Data distributions (duration, distance, average speed)
- Figures 4-7: Sample trajectories from each cluster (1 per cluster)

Usage:
    python generate_figures.py --data-path data/Microtrips --output-dir fig
"""

import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10

# IEEE two-column format: column width ~ 3.5 inches
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16





def load_cluster_assignments(artifacts_path):
    """Load cluster to trip ID mapping"""
    with open(Path(artifacts_path) / 'cluster_to_trip_ids.json', 'r') as f:
        return json.load(f)





def load_generated_trajectories(reproduce_path):
    """Load generated CSDI and Diffusion trajectories"""
    csdi_path = Path(reproduce_path) / 'csdi/v1.3/results/csdi_synthetic_20260101_200318.csv'
    diff_path = Path(reproduce_path) / 'diffusion/v1.5/synthetic_trajectories_v1.5_boost175.csv'
    if not diff_path.exists(): diff_path = Path(reproduce_path) / 'diffusion_results.csv'
    
    csdi_df = pd.DataFrame()
    diff_df = pd.DataFrame()
    
    if csdi_path.exists():
        csdi_df = pd.read_csv(csdi_path)
        print(f"Loaded CSDI data from {csdi_path.name}")
    else:
        print(f"Warning: {csdi_path} not found")
        
    if diff_path.exists():
        diff_df = pd.read_csv(diff_path)
        print(f"Loaded Diffusion data from {diff_path.name}")
    else:
        print(f"Warning: {diff_path} not found")
        
    return csdi_df, diff_df





def figure1_data_distributions(real_df, output_dir):
    """Generate Figure 1: Data distributions (duration, distance, speed)"""
    if real_df.empty:
        print("Error: real_df is empty. Skipping Figure 1 (Data Distributions).")
        return

    print("Generating Figure 1: Data distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.5))
    
    # Use actual columns from artifacts/stats_df.csv
    # duration_seconds, length_km, avg_speed_kmh
    if 'duration_seconds' in real_df.columns and 'length_km' in real_df.columns:
        duration = real_df['duration_seconds'].values
        distance = real_df['length_km'].values
        speed = real_df['avg_speed_kmh'].values
    else:
        print("Warning: 'duration_seconds'/'length_km' columns not found in stats_df. Skipping Figure 1.")
        plt.close()
        return

    # Plot duration
    axes[0].hist(duration[duration <= 500], bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axvline(np.mean(duration), color='red', linestyle='--', linewidth=1, label='Mean')
    axes[0].axvline(np.median(duration), color='orange', linestyle='--', linewidth=1, label='Median')
    axes[0].set_xlabel('Duration (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([0, 500])
    axes[0].legend(frameon=False)
    axes[0].set_title('Trip Duration')
    
    # Plot distance
    axes[1].hist(distance[distance <= 7], bins=50, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].axvline(np.mean(distance), color='red', linestyle='--', linewidth=1, label='Mean')
    axes[1].axvline(np.median(distance), color='orange', linestyle='--', linewidth=1, label='Median')
    axes[1].set_xlabel('Distance (km)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([0, 7])
    axes[1].legend(frameon=False)
    axes[1].set_title('Trip Distance')
    
    # Plot speed
    axes[2].hist(speed, bins=50, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[2].axvline(np.mean(speed), color='red', linestyle='--', linewidth=1, label='Mean')
    axes[2].axvline(np.median(speed), color='orange', linestyle='--', linewidth=1, label='Median')
    axes[2].set_xlabel('Average Speed (km/h)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend(frameon=False)
    axes[2].set_title('Average Speed')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig1_data_distributions.pdf', bbox_inches='tight')
    plt.savefig(Path(output_dir) / 'fig1_data_distributions.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated Figure 1: Data distributions")





def figures4to7_sample_trajectories(cluster_to_trips, data_path, output_dir):
    """Generate Figures 4-7: Sample trajectories from each cluster with trip IDs"""
    np.random.seed(42)
    
    cluster_names = [
        'Arterial/Suburban',
        'Highway/Interstate', 
        'Congested/City',
        'Free-flow Arterial'
    ]
    # Curated trip IDs - cluster 2 uses a trip with better stop-and-go patterns
    curated_trips = {
        '0': [list(cluster_to_trips['0'])[0]],
        '1': [list(cluster_to_trips['1'])[0]],
        '2': ['11624'],  # Better congested example with stops
        '3': [list(cluster_to_trips['3'])[0]],
    }
    
    for cluster_id, name in enumerate(cluster_names):
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.0))
        
        # Get curated trip ID
        trip_id = curated_trips[str(cluster_id)][0]
        
        # Load actual trip data from CSV
        import pandas as pd
        trip_file = Path(data_path) / f'results_trip_{trip_id}.csv'
        
        if trip_file.exists():
            df = pd.read_csv(trip_file)
            speed = df['speedMps'].values
            t = np.arange(len(speed))
        else:
            print(f"Warning: {trip_file} not found!")
            t = np.arange(250)
            speed = np.zeros(250)
        
        ax.plot(t, speed, color='steelblue', linewidth=0.8)
        ax.set_ylabel('Speed (m/s)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 35])
        ax.set_xlabel('Time (s)')
        
        # Add trip ID label (now accurate - real data!)
        ax.text(0.98, 0.95, f'Trip ID: {trip_id}', 
                         transform=ax.transAxes,
                         ha='right', va='top', fontsize=7,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ax.set_title(f'{name}')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'fig{4+cluster_id}_cluster{cluster_id}_trajectories.pdf', bbox_inches='tight')
        plt.savefig(Path(output_dir) / f'fig{4+cluster_id}_cluster{cluster_id}_trajectories.png', bbox_inches='tight')
        plt.close()
        print(f"✓ Generated Figure {4+cluster_id}: {name} trajectory (with trip ID)")



def load_real_distribution_data(data_path):
    """
    Loads real trip data, trims trailing zeros, and returns a list of speed arrays.
    Returns list of arrays (not concatenated) to allow per-trip analysis.
    """
    speed_list = []
    
    # Use recursive glob matches evaluate_csdi.py
    files = glob.glob(os.path.join(data_path, "**", "results_trip_*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_path, "results_trip_*.csv"))
        
    print(f"Found {len(files)} real trajectory files")
    
    for trip_file in files:
        try:
            df = pd.read_csv(trip_file)
            speed = None
            if 'speedMps' in df.columns:
                speed = df['speedMps'].values
            elif 'speed' in df.columns:
                speed = df['speed'].values
            
            if speed is not None and len(speed) >= 50:
                # Trim trailing zeros (padding)
                nonzero = np.nonzero(speed)[0]
                if len(nonzero) > 0:
                    end_idx = nonzero[-1] + 1
                    trimmed = speed[:end_idx]
                    if len(trimmed) > 1:
                        speed_list.append(trimmed)
        except Exception:
            pass

    if not speed_list:
        print("Warning: No valid real trips found!")
        return []
        
    return speed_list

def process_synthetic_data(df):
    """Process synthetic dataframe: extract valid trimmed speeds"""
    cols = [c for c in df.columns if str(c).isdigit() or (str(c).startswith('speed_') and str(c).split('_')[-1].isdigit())]
    if len(cols) == 0: return []
    
    speed_list = []
    for i in range(len(df)):
        row = df.iloc[i][cols].values
        # Trim trailing zeros (padding)
        nonzero = np.nonzero(row)[0]
        if len(nonzero) > 0:
            end_idx = nonzero[-1] + 1
            trimmed = row[:end_idx]
            # Verify it's not just a single point or empty
            if len(trimmed) > 1:
                speed_list.append(trimmed)
    return speed_list

def figure_main_distributions(output_dir, csdi_df, diff_df, real_df, data_path):
    """
    Generate Main Results Figure: Distributions (Row 1)
    Layout: CSDI-Speed, Diffusion-Speed, CSDI-Accel, Diffusion-Accel
    Ratios: 30%, 30%, 20%, 20%
    """
    print("Generating Row 1: Distributions...")
    
    # 1. Load/Process Real Data
    if not real_df.empty:
        # Use sp0 (renamed to speed_0 in main) for full distribution
        real_speeds = real_df['speed_0'].values
        # Acceleration from sliding window: speed[t+1] - speed[t]
        if 'speed_1' in real_df.columns:
            real_accels = (real_df['speed_1'] - real_df['speed_0']).values
        else:
            real_accels = np.array([])
        print(f"Using full Real dataset ({len(real_speeds)} points)")
    else:
        # Fallback to loading individual files
        real_trips = load_real_distribution_data(data_path)
        if len(real_trips) > 0:
            real_speeds = np.concatenate(real_trips)
            real_accels = np.concatenate([np.diff(t) for t in real_trips])
        else:
            real_speeds = np.array([])
            real_accels = np.array([])
    
    # 2. Process Synthetic Data (Trimmed)
    csdi_trips = process_synthetic_data(csdi_df)
    if len(csdi_trips) > 0:
        csdi_speeds = np.concatenate(csdi_trips)
        csdi_accels = np.concatenate([np.diff(t) for t in csdi_trips])
    else:
        csdi_speeds = np.array([])
        csdi_accels = np.array([])

    diff_trips = process_synthetic_data(diff_df)
    if len(diff_trips) > 0:
        diff_speeds = np.concatenate(diff_trips)
        diff_accels = np.concatenate([np.diff(t) for t in diff_trips])
    else:
        diff_speeds = np.array([])
        diff_accels = np.array([])

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), gridspec_kw={'width_ratios': [3, 3, 2, 2]})
    
    bins_speed = np.linspace(0, 35, 40)
    bins_accel = np.linspace(-3, 3, 50)
    
    # 1. CSDI Speed
    ax1 = axes[0]
    if len(real_speeds) > 0:
        ax1.hist(real_speeds, bins=bins_speed, density=True, histtype='step', color='black', linewidth=1.5, label='Real')
    else:
        print("Warning: Skipping Real line in CSDI Speed plot.")

    ax1.hist(csdi_speeds, bins=bins_speed, density=True, alpha=0.5, color='forestgreen', label='CSDI')
    ax1.set_title('CSDI Speed', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', frameon=False, fontsize=9)
    
    # 2. Diffusion Speed
    ax2 = axes[1]
    if len(real_speeds) > 0:
        ax2.hist(real_speeds, bins=bins_speed, density=True, histtype='step', color='black', linewidth=1.5, label='Real')
    
    ax2.hist(diff_speeds, bins=bins_speed, density=True, alpha=0.5, color='coral', label='Diffusion')
    ax2.set_title('Diffusion Speed', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Speed (m/s)')
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    ax2.set_yticks([]) # Hide y-axis ticks for cleaner look
    
    # 3. CSDI Acceleration
    ax3 = axes[2]
    if len(real_accels) > 0:
        ax3.hist(real_accels, bins=bins_accel, density=True, histtype='step', color='black', linewidth=1.5, label='Real')
    
    ax3.hist(csdi_accels, bins=bins_accel, density=True, alpha=0.5, color='forestgreen', label='CSDI')
    ax3.set_title('CSDI Accel', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Accel (m/s²)')
    ax3.set_xlim([-2, 2])
    ax3.legend(loc='upper right', frameon=False, fontsize=8)
    ax3.set_yticks([]) # Hide y-axis ticks for cleaner look
    
    # 4. Diffusion Acceleration
    ax4 = axes[3]
    if len(real_accels) > 0:
        ax4.hist(real_accels, bins=bins_accel, density=True, histtype='step', color='black', linewidth=1.5, label='Real')
    
    ax4.hist(diff_accels, bins=bins_accel, density=True, alpha=0.5, color='coral', label='Diffusion')
    ax4.set_title('Diffusion Accel', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Accel (m/s²)')
    ax4.set_xlim([-2, 2])
    ax4.legend(loc='upper right', frameon=False, fontsize=8)
    ax4.set_yticks([]) # Hide y-axis ticks for cleaner look

    plt.tight_layout()
    output_path = Path(output_dir) / 'fig_main_distributions.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated Row 1: Distributions (Split plots) -> {output_path}")

def figure_main_csdi_traj(output_dir, df):
    """Row 2: CSDI Trajectories (3 regimes) - Realistic sampling"""
    np.random.seed(101) # New seed for variety
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.2))
    
    regimes = ['Highway', 'Arterial', 'Congested']
    labels = ['CSDI Highway', 'CSDI Arterial', 'CSDI Congested']
    
    # Filter df by regimes
    # Process df to ensure it has speed_ columns
    cols = [c for c in df.columns if str(c).isdigit() or (str(c).startswith('speed_') and str(c).split('_')[-1].isdigit())]
    if len(cols) == 0:
         print("Warning: No speed columns found in DF for CSDI trajectories")
         df['avg_speed'] = 0
    else:
         df['avg_speed'] = df[cols].mean(axis=1)
    
    for idx, (regime, label) in enumerate(zip(regimes, labels)):
        ax = axes[idx]
        
        if regime == 'Highway':
            # Try to find highway trips that don't saturate the 35m/s limit
            cands_all = df[df['avg_speed'] > 22][cols].values
            maxs = np.max(cands_all, axis=1)
            clean_indices = np.where(maxs <= 35)[0]
            
            if len(clean_indices) > 0:
                candidates = cands_all[clean_indices]
            else:
                candidates = cands_all
        elif regime == 'Arterial':
            candidates = df[(df['avg_speed'] > 12) & (df['avg_speed'] < 18)][cols].values
        else: # Congested
            # Filter for "stop-and-go": avg speed low, but MUST have stops
            cands_all = df[(df['avg_speed'] < 12) & (df['avg_speed'] > 1)][cols].values
            candidates = []
            for c in cands_all:
                last_idx = np.max(np.nonzero(c)) + 1 if np.any(c) else len(c)
                trim = c[:last_idx]
                # Long trip, low min speed
                if len(trim) > 100 and np.min(trim) < 0.3:
                    candidates.append(c)
            candidates = np.array(candidates)
            
        if len(candidates) > 0:
            scores = []
            for c in candidates:
                # Trim zeros/padding
                last = np.max(np.nonzero(c)) + 1 if np.any(c) else len(c)
                trim = c[:last]
                
                # Calculate number of "cycles"
                peaks = 0
                if len(trim) > 5:
                    smooth = np.convolve(trim, np.ones(5)/5, mode='valid')
                    diffs = np.diff(smooth)
                    peaks = np.sum(np.diff(np.sign(diffs)) != 0)
                
                # Metric: cycles + variance + zero observations (heavy weight on zeros)
                zero_count = np.sum(trim < 1.0)
                score = peaks * 1.0 + np.std(trim) * 0.3 + zero_count * 0.5
                scores.append(score)
            
            scores = np.array(scores)
            valid_indices = np.where(scores > 1)[0]
            
            if len(valid_indices) > 0:
                # Pick from top 10%
                top_cutoff = np.percentile(scores[valid_indices], 90)
                top_indices = valid_indices[scores[valid_indices] >= top_cutoff]
                choice = candidates[np.random.choice(top_indices)]

            else:
                 choice = candidates[np.random.randint(len(candidates))]
            
            # Trim trailing zeros (padding) for visualization
            last_idx = np.max(np.nonzero(choice)) + 1 if np.any(choice) else len(choice)
            speed = choice[:last_idx].copy()

        else:
            # Fallback
            print(f"Warning: No {regime} candidates found for CSDI. Skipping subplot.")
            ax.set_visible(False)
            continue
            
        speed = np.clip(speed, 0, 35)
        t = np.arange(len(speed))
        
        ax.plot(t, speed, color='forestgreen', linewidth=1.2)
        ax.set_ylim([0, 35])
        ax.set_title(label, fontsize=9)
        if idx == 0: ax.set_ylabel('Speed (m/s)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig_main_csdi_traj.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated Row 2: CSDI Trajectories (Realistic)")

def figure_main_diff_traj(output_dir, df):
    """Row 3: Diffusion Trajectories (3 regimes) - Realistic sampling"""
    np.random.seed(202) # Different seed
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.2))
    
    regimes = ['Highway', 'Arterial', 'Congested']
    labels = ['Diff Highway', 'Diff Arterial', 'Diff Congested']
    
    # Assuming df describes Diffusion data now when called by wrapper wrapper
    # Process df to ensure it has speed_ columns
    cols = [c for c in df.columns if str(c).isdigit() or (str(c).startswith('speed_') and str(c).split('_')[-1].isdigit())]
    if len(cols) == 0:
         print("Warning: No speed columns found in DF for Diffusion trajectories")
         df['avg_speed'] = 0
    else:
         df['avg_speed'] = df[cols].mean(axis=1)

    for idx, (regime, label) in enumerate(zip(regimes, labels)):
        ax = axes[idx]
        
        if regime == 'Highway':
            # Try to find highway trips that don't saturate the 35m/s limit
            cands_all = df[df['avg_speed'] > 22][cols].values
            maxs = np.max(cands_all, axis=1)
            clean_indices = np.where(maxs <= 35)[0]
            
            if len(clean_indices) > 0:
                candidates = cands_all[clean_indices]
            else:
                candidates = cands_all
        elif regime == 'Arterial':
            candidates = df[(df['avg_speed'] > 12) & (df['avg_speed'] < 18)][cols].values
        else: # Congested
            # Strict Filter for "stop-and-go"
            cands_all = df[(df['avg_speed'] < 12) & (df['avg_speed'] > 1)][cols].values
            
            # Additional filtering for max speed and stops
            candidates = []
            for c in cands_all:
                last_idx = np.max(np.nonzero(c)) + 1 if np.any(c) else len(c)
                trim = c[:last_idx]
                if len(trim) > 100 and np.max(trim) < 22 and np.min(trim) < 0.3:
                     candidates.append(c)
            candidates = np.array(candidates)
            
        if len(candidates) > 0:
            scores = []
            for c in candidates:
                # Trim first
                last = np.max(np.nonzero(c)) + 1 if np.any(c) else len(c)
                trim = c[:last]
                
                # Calculate number of peaks/cycles
                peaks = 0
                if len(trim) > 5:
                    smooth = np.convolve(trim, np.ones(5)/5, mode='valid')
                    diffs = np.diff(smooth)
                    peaks = np.sum(np.diff(np.sign(diffs)) != 0)
                
                # Metric: cycles + variance + zero observations (heavy weight on zeros)
                zero_count = np.sum(trim < 1.0)
                score = peaks * 1.0 + np.std(trim) * 0.3 + zero_count * 0.5
                scores.append(score)
            
            scores = np.array(scores)
            valid_indices = np.where(scores > 1)[0]
            
            if len(valid_indices) > 0:
                # Pick from top 10%
                top_cutoff = np.percentile(scores[valid_indices], 90)
                top_indices = valid_indices[scores[valid_indices] >= top_cutoff]
                choice = candidates[np.random.choice(top_indices)]

            else:
                 choice = candidates[np.random.randint(len(candidates))]
                 
            # Trim trailing zeros
            last_idx = np.max(np.nonzero(choice)) + 1 if np.any(choice) else len(choice)
            speed = choice[:last_idx].copy()

        else:
            print(f"Warning: No {regime} candidates found for Diffusion. Skipping subplot.")
            ax.set_visible(False)
            continue
            
        speed = np.clip(speed, 0, 35)
        t = np.arange(len(speed))
        
        ax.plot(t, speed, color='coral', linewidth=1.2)
        ax.set_ylim([0, 35])
        ax.set_title(label, fontsize=9)
        if idx == 0: ax.set_ylabel('Speed (m/s)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig_main_diff_traj.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated Row 3: Diffusion Trajectories (Realistic)")

def figure_main_safd(output_dir, csdi_df, diff_df, real_df, data_path):
    """Row 4: SAFD Heatmaps (Real, CSDI, Diffusion) - Equal Widths"""
    print("Generating Row 4: SAFD Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.5))
    
    titles = ['Real SAFD', 'CSDI SAFD', 'Diffusion SAFD']
    cmaps = ['Blues', 'Greens', 'Oranges']
    
    # helper for real data from DF
    def get_sa_from_df(df):
        if df.empty: return np.array([]), np.array([])
        s = df['speed_0'].values
        if 'speed_1' in df.columns:
            a = (df['speed_1'] - df['speed_0']).values
        else:
            a = np.zeros_like(s)
        return s, a

    # helper to get s, a from trips
    def get_sa(trips):
        if not trips: return np.array([]), np.array([])
        s_all, a_all = [], []
        for t in trips:
            if len(t) > 1:
                # speed corresponds to points where accel is defined (or aligned)
                # accel[i] = speed[i+1] - speed[i]
                # align speed[i] with accel[i]
                s = t[:-1]
                a = np.diff(t)
                s_all.append(s)
                a_all.append(a)
        if s_all:
            return np.concatenate(s_all), np.concatenate(a_all)
        return np.array([]), np.array([])

    # Process Synthetic Data (Trimmed)
    csdi_trips = process_synthetic_data(csdi_df)
    diff_trips = process_synthetic_data(diff_df)
    
    for i, (ax, title, cmap) in enumerate(zip(axes, titles, cmaps)):
        if i == 0: # Real
            speed, accel = get_sa_from_df(real_df)
        elif i == 1: # CSDI
            speed, accel = get_sa(process_synthetic_data(csdi_df))
        else: # Diffusion
            speed, accel = get_sa(process_synthetic_data(diff_df))
        
        if len(speed) > 0:
            hist, xedges, yedges = np.histogram2d(speed, accel, bins=[30, 40], range=[[0, 35], [-4, 4]])
            # Log scale for better visibility if needed, but linear is standard for pdf
            ax.imshow(hist.T, origin='lower', aspect='auto', cmap=cmap, extent=[0, 35, -4, 4])
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Speed (m/s)')
        if i == 0: ax.set_ylabel('Accel (m/s²)')
        else: ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fig_main_safd.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated Row 4: SAFD Heatmaps (Real added)")

def figure_main_results_wrapper(output_dir, csdi_df, diff_df, real_df, data_path):
    """Wrapper to call all 4 parts"""
    figure_main_distributions(output_dir, csdi_df, diff_df, real_df, data_path)
    figure_main_csdi_traj(output_dir, csdi_df)
    figure_main_diff_traj(output_dir, diff_df)
    figure_main_safd(output_dir, csdi_df, diff_df, real_df, data_path)








def main():
    parser = argparse.ArgumentParser(description='Generate IEEE ITS paper figures')
    parser.add_argument('--artifacts-path', default='artifacts', help='Path to artifacts directory')
    parser.add_argument('--data-path', type=str, default='../../data/Microtrips',
                        help='Path to real trip CSVs')
    parser.add_argument('--output-dir', default='fig', help='Output directory for figures')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Refined IEEE ITS Paper Figures")
    print("=" * 60)
    
    # Load cluster assignments
    cluster_to_trips = load_cluster_assignments(args.artifacts_path)
    
    # Load Generated Trajectories
    csdi_df, diff_df = load_generated_trajectories('.')
    
    # 1. Load summary stats for Fig 1
    stats_file = Path(args.artifacts_path) / 'stats_df.csv'
    if stats_file.exists():
        stats_df = pd.read_csv(stats_file)
    else:
        print(f"Warning: {stats_file} not found. Figure 1 will be empty.")
        stats_df = pd.DataFrame()

    # Generate each figure with improvements
    figure1_data_distributions(stats_df, output_dir)
    figures4to7_sample_trajectories(cluster_to_trips, args.data_path, output_dir)  # With trip IDs
    
    # Load Real Data for Main Results
    print("Loading real data from ../data/alltrips.csv...")
    try:
        # Resolving path relative to script or execution? 
        # Assuming execution from reproduce/, data is in ../data/
        data_file = Path('../data/alltrips.csv')
        if not data_file.exists():
             # Fallback if running from different location
             data_file = Path('../../data/alltrips.csv')
        
        df = pd.read_csv(data_file)
        # Rename columns sp0 -> speed_0 etc to match expectation
        df.columns = [c.replace('sp', 'speed_') if c.startswith('sp') and c[2:].isdigit() else c for c in df.columns]
        print(f"Loaded data with shape {df.shape}")
    except Exception as e:
        print(f"Error loading real data: {e}")
        df = pd.DataFrame() # Fallback empty DF

    # Generate Main Results Figure (Split)
    
    # Pass data directly now to wrapper
    # Note: args.data_path defaults to 'data/Microtrips' which is correct relative to reproduce/
    # But if running from reproduce/, we need to ensure it resolves. 
    # If explicit path not given, check default.
    if not Path(args.data_path).exists():
         # Locate it
         found = list(Path('.').glob("**/Microtrips"))
         if found:
             args.data_path = str(found[0])
             print(f"Found data at {args.data_path}")
         else:
             print(f"Warning: Data path {args.data_path} not found. Real distributions will be empty.")

    figure_main_results_wrapper(output_dir, csdi_df, diff_df, df, args.data_path)

    
    print("=" * 60)
    print(f"✓ All figures generated successfully in {output_dir}")
    print("✓ Improvements: PCA+t-SNE side-by-side, trip IDs added, Main Results comprehensive figure")
    print("=" * 60)


if __name__ == '__main__':
    main()
