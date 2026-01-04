import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import sys
from datetime import datetime
import glob
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# Add shared metrics module to path
# Priority: Local (Snapshot) > Shared (Dev)
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'evaluation_metrics.py')):
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'sdv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'sdv')
    ]
    for p in possible_paths:
        if os.path.exists(p):
            sys.path.insert(0, p)
            break
try:
    from evaluation_metrics import (
        safd_wasserstein_distance,
        compute_mmd,
        discriminative_score,
        predictive_score_tstr,
        boundary_violation_rate,
        compute_ldlj as shared_ldlj
    )
    HAS_SHARED_METRICS = True
except ImportError:
    HAS_SHARED_METRICS = False
    print("Warning: Could not import shared evaluation_metrics. Some metrics will be unavailable.")

def compute_kinematics(speeds):
    """
    speeds: list of arrays or list of lists (m/s)
    Returns: flattened arrays of speed, accel, jerk
    """
    all_speeds = []
    all_accels = []
    all_jerks = []
    
    for v in speeds:
        v = np.array(v)
        # remove padding (trailing zeros) - assuming trips don't end with long zeros in valid data
        # but synthetic might have it.
        # Simple trim for evaluation: trim trailing zeros if they are long
        # For now, take as is, but be mindful of padding.
        
        a = np.diff(v, prepend=v[0]) # Accel
        j = np.diff(a, prepend=a[0]) # Jerk
        
        all_speeds.extend(v)
        all_accels.extend(a)
        all_jerks.extend(j)
        
    return np.array(all_speeds), np.array(all_accels), np.array(all_jerks)

def compute_safd(speeds, accels, bins=50):
    """
    Speed-Acceleration Frequency Distribution (2D Histogram)
    """
    heatmap, xedges, yedges = np.histogram2d(speeds, accels, bins=bins, range=[[0, 35], [-5, 5]], density=True)
    return heatmap

def compute_vsp(v, a):
    # VSP = v * (1.1 * a + 9.81 * sin(0) + 0.132) + 0.000302 * v^3
    # grade = 0
    return v * (1.1 * a + 0.132) + 0.000302 * (v**3)

def compute_ldlj(speeds, dt=1.0):
    """
    Log Dimensionless Jerk (LDLJ) - Measure of smoothness.
    Lower (more negative) is better? No, typically closer to 0 (less jerk) relative to duration.
    Actually, LDLJ = -log(|jerk_norm|).
    """
    ldljs = []
    for v in speeds:
        if len(v) < 3: continue
        # a = dv/dt, j = da/dt
        a = np.diff(v) / dt
        j = np.diff(a) / dt
        
        # Dimensionless Jerk
        duration = len(v) * dt
        v_max = np.max(np.abs(v)) + 1e-6
        j_norm = np.sum(j**2) * (duration**3) / (v_max**2)
        
        if j_norm > 0:
            ldljs.append(-np.log(j_norm))
    return np.mean(ldljs) if ldljs else 0.0

def compute_safd_wasserstein(real_v, real_a, syn_v, syn_a, bins=50):
    """
    Wasserstein distance between 2D SAFD distributions.
    Simplified as WD between flattened normalized histograms.
    """
    h_real = compute_safd(real_v, real_a, bins=bins)
    h_syn = compute_safd(syn_v, syn_a, bins=bins)
    
    # Flatten and treat as 1D distributions for WD
    return wasserstein_distance(h_real.flatten(), h_syn.flatten())

def check_boundary_violations(speeds, threshold=0.1):
    """
    Percentage of trips that do NOT start/end at 0.
    """
    violations = 0
    for v in speeds:
        if abs(v[0]) > threshold or abs(v[-1]) > threshold:
            violations += 1
    return (violations / len(speeds)) * 100 if speeds else 0.0

def evaluate(syn_csv="data/synthetic_trajectories.csv", real_pkl="data/real_data_stats.pkl", suffix=""):
    print("--- Starting Evaluation ---")
    
    # Generate timestamp
    now = datetime.now()
    timestamp = now.strftime("%b%d_%I%M%p").lower() # e.g. jan03_1112am
    
    # Ensure report directory exists
    os.makedirs("report", exist_ok=True)
    
    # 1. Load Synthetic
    if not os.path.exists(syn_csv):
        print(f"Warning: {syn_csv} not found.")
        # Try to find latest timestamped CSV in data/
        csv_pattern = "data/synthetic_trajectories_*.csv"
        found_csvs = sorted(glob.glob(csv_pattern))
        if found_csvs:
            syn_csv = found_csvs[-1]
            print(f"Using latest found CSV: {syn_csv}")
        else:
            print(f"Error: No synthetic CSV found matching {csv_pattern}")
            return

    df_syn = pd.read_csv(syn_csv)
    # Exclude target cols
    traj_cols = [c for c in df_syn.columns if c not in ["target_len", "target_spd"]]
    syn_trajs = df_syn[traj_cols].values
    
    # Clean synthetic: trim to target length to avoid padding processing
    syn_speeds_clean = []
    for i, row in df_syn.iterrows():
        l = int(row["target_len"])
        s = row.values[:l]
        syn_speeds_clean.append(s)

    syn_v_flat, syn_a_flat, syn_j_flat = compute_kinematics(syn_speeds_clean)
    
    # 2. Load Real
    with open(real_pkl, "rb") as f:
        real_stats = pickle.load(f)
    
    real_speeds_list = real_stats["speeds"]
    real_v_flat, real_a_flat, real_j_flat = compute_kinematics(real_speeds_list)
    
    # 3. VSP
    syn_vsp = compute_vsp(syn_v_flat, syn_a_flat)
    real_vsp = compute_vsp(real_v_flat, real_a_flat)
    
    # 4. Metrics
    print("Calculating metrics...")
    
    # Wasserstein (Earth Mover's Distance) for distributions
    wd_speed = wasserstein_distance(real_v_flat, syn_v_flat)
    wd_accel = wasserstein_distance(real_a_flat, syn_a_flat)
    wd_vsp = wasserstein_distance(real_vsp, syn_vsp)
    
    # New Advanced Metrics
    ldlj_real = compute_ldlj(real_speeds_list)
    ldlj_syn = compute_ldlj(syn_speeds_clean)
    
    if HAS_SHARED_METRICS:
        wd_safd = safd_wasserstein_distance(real_speeds_list, syn_speeds_clean)
        boundary_info = boundary_violation_rate(syn_speeds_clean, threshold=0.1)
        boundary_violation = boundary_info["boundary_any_violation_rate"] * 100
        
        print("Calculating MMD (this may take a few seconds)...")
        mmd_val = compute_mmd(real_speeds_list, syn_speeds_clean, n_samples=200)
        
        print("Calculating Discriminative Score...")
        disc_results = discriminative_score(real_speeds_list, syn_speeds_clean)
        disc_val = disc_results.get("discriminative_score", 0.0)
        
        print("Calculating TSTR MAE...")
        tstr_results = predictive_score_tstr(real_speeds_list, syn_speeds_clean, n_samples=500)
        tstr_mae = tstr_results.get("tstr_mae", 0.0)
    else:
        wd_safd = compute_safd_wasserstein(real_v_flat, real_a_flat, syn_v_flat, syn_a_flat)
        boundary_violation = check_boundary_violations(syn_speeds_clean)
        mmd_val = 0.0
        disc_val = 0.0
        tstr_mae = 0.0
    
    # KS Test
    ks_vsp = ks_2samp(real_vsp, syn_vsp)

    # Summary Stats
    real_mean_v = np.mean([np.mean(t) for t in real_speeds_list])
    syn_mean_v = np.mean([np.mean(t) for t in syn_speeds_clean])
    real_max_v = np.max(real_v_flat)
    syn_max_v = np.max(syn_v_flat)
    real_mean_dur = np.mean([len(t) for t in real_speeds_list])
    syn_mean_dur = np.mean([len(t) for t in syn_speeds_clean])

    print("-" * 40)
    print(f"{'Metric':<25} {'Value':<10}")
    print("-" * 40)
    print(f"Wasserstein Speed:       {wd_speed:.4f}")
    print(f"Wasserstein Accel:       {wd_accel:.4f}")
    print(f"Wasserstein VSP:         {wd_vsp:.4f}")
    print(f"Wasserstein SAFD (2D):   {wd_safd:.4f}")
    print(f"MMD:                     {mmd_val:.4f}")
    print(f"Discriminative Score:    {disc_val:.4f}")
    print(f"TSTR MAE:                {tstr_mae:.4f}")
    print(f"LDLJ (Smoothness) Real:  {ldlj_real:.4f}")
    print(f"LDLJ (Smoothness) Syn:   {ldlj_syn:.4f}")
    print(f"Boundary Violation Rate: {boundary_violation:.2f}%")
    print(f"KS Stat VSP:             {ks_vsp.statistic:.4f} (p={ks_vsp.pvalue:.4f})")
    print("-" * 40)
    
    print("\nSummary Comparison:")
    print(f"{'Stat':<20} {'Real':<10} {'Syn':<10}")
    print(f"{'Mean Speed (m/s)':<20} {real_mean_v:<10.2f} {syn_mean_v:<10.2f}")
    print(f"{'Max Speed (m/s)':<20} {real_max_v:<10.2f} {syn_max_v:<10.2f}")
    print(f"{'Mean Duration (s)':<20} {real_mean_dur:<10.1f} {syn_mean_dur:<10.1f}")
    print("-" * 40)
    
    # 5. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Speed Dist
    axes[0,0].hist(real_v_flat, bins=50, density=True, alpha=0.5, label='Real', color='blue')
    axes[0,0].hist(syn_v_flat, bins=50, density=True, alpha=0.5, label='Syn', color='orange')
    axes[0,0].set_title(f"Speed Dist (WD={wd_speed:.2f})")
    axes[0,0].legend()
    
    # Accel Dist
    axes[0,1].hist(real_a_flat, bins=50, density=True, alpha=0.5, label='Real', color='blue', range=(-5, 5))
    axes[0,1].hist(syn_a_flat, bins=50, density=True, alpha=0.5, label='Syn', color='orange', range=(-5, 5))
    axes[0,1].set_title(f"Accel Dist (WD={wd_accel:.2f})")
    
    # VSP Dist
    axes[0,2].hist(real_vsp, bins=50, density=True, alpha=0.5, label='Real', color='blue', range=(-30, 30))
    axes[0,2].hist(syn_vsp, bins=50, density=True, alpha=0.5, label='Syn', color='orange', range=(-30, 30))
    axes[0,2].set_title(f"VSP Dist (WD={wd_vsp:.2f})")

    # SAFD Real
    h_real = compute_safd(real_v_flat, real_a_flat)
    axes[1,0].imshow(np.rot90(h_real), extent=[0,35,-5,5], aspect='auto', cmap='jet')
    axes[1,0].set_title("Real SAFD")
    axes[1,0].set_xlabel("Speed")
    axes[1,0].set_ylabel("Accel")
    
    # SAFD Syn
    h_syn = compute_safd(syn_v_flat, syn_a_flat)
    axes[1,1].imshow(np.rot90(h_syn), extent=[0,35,-5,5], aspect='auto', cmap='jet')
    axes[1,1].set_title("Synthetic SAFD")
    
    # Trajectory Samples
    for i in range(min(10, len(syn_speeds_clean))):
        axes[1,2].plot(syn_speeds_clean[i], alpha=0.7)
    axes[1,2].set_title("Generated Samples")
    
    plt.tight_layout()
    plot_name = f"report/evaluation_plots{suffix}_{timestamp}.png"
    plt.savefig(plot_name)
    print(f"Saved {plot_name}")
    
    # Save Report
    report_name = f"report/evaluation_report{suffix}_{timestamp}.md"
    with open(report_name, "w") as f:
        f.write(f"# Evaluation Report (Suffix: {suffix}, Timestamp: {timestamp})\n\n")
        f.write("## Distribution Metrics\n")
        f.write(f"- WD Speed:       {wd_speed:.4f}\n")
        f.write(f"- WD Accel:       {wd_accel:.4f}\n")
        f.write(f"- WD VSP:         {wd_vsp:.4f}\n")
        f.write(f"- WD SAFD (2D):   {wd_safd:.4f}\n")
        f.write(f"- MMD:            {mmd_val:.4f}\n")
        f.write(f"- Discriminative Score: {disc_val:.4f}\n")
        f.write(f"- TSTR MAE:       {tstr_mae:.4f}\n")
        f.write(f"- KS VSP:         {ks_vsp.statistic:.4f}\n\n")
        
        f.write("## Kinematic Metrics\n")
        f.write(f"- LDLJ (Smoothness) Real: {ldlj_real:.4f}\n")
        f.write(f"- LDLJ (Smoothness) Syn:  {ldlj_syn:.4f}\n")
        f.write(f"- Boundary Violation Rate: {boundary_violation:.2f}%\n\n")
        
        f.write("## Summary Statistics\n")
        f.write("| Metric | Real | Synthetic |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Mean Speed (m/s) | {real_mean_v:.2f} | {syn_mean_v:.2f} |\n")
        f.write(f"| Max Speed (m/s) | {real_max_v:.2f} | {syn_max_v:.2f} |\n")
        f.write(f"| Mean Duration (s) | {real_mean_dur:.1f} | {syn_mean_dur:.1f} |\n")
    print(f"Saved {report_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=str, default="data/synthetic_trajectories.csv", help="Path to synthetic CSV")
    parser.add_argument("--real", type=str, default="data/real_data_stats.pkl", help="Path to real stats PKL")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output files")
    args = parser.parse_args()
    
    evaluate(syn_csv=args.synthetic, real_pkl=args.real, suffix=args.suffix)
