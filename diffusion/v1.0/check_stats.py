
import pandas as pd
import numpy as np
import glob
import os
import pickle

def analyze_stats():
    print("--- Analyzing Real Data ---")
    # Load Real Data Stats if available
    stats_path = 'data/real_data_stats.pkl'
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            print(f"Loaded Real Stats from {stats_path}")
    
    # Manually check a few files to verify
    files = glob.glob("/projects/vsokolov/svtrip/data/Microtrips/*.csv")[:100]
    all_speeds = []
    for f in files:
        df = pd.read_csv(f)
        if 'speedMps' in df.columns:
            all_speeds.append(df['speedMps'].values)
    
    if all_speeds:
        flat_speeds = np.concatenate(all_speeds)
        print(f"Real Data (Sample 100 files):")
        print(f"  Min: {flat_speeds.min():.4f}")
        print(f"  Max: {flat_speeds.max():.4f}")
        print(f"  Mean: {flat_speeds.mean():.4f}")
        print(f"  Std: {flat_speeds.std():.4f}")
        
        # Check for outliers
        p95 = np.percentile(flat_speeds, 95)
        p99 = np.percentile(flat_speeds, 99)
        print(f"  95th Percentile: {p95:.4f}")
        print(f"  99th Percentile: {p99:.4f}")
        
    all_accels = []
    for f in files:
        df = pd.read_csv(f)
        if 'accelMps2' in df.columns:
            all_accels.append(df['accelMps2'].values)
    
    if all_accels:
        flat_accels = np.concatenate(all_accels)
        print(f"Real Accel:")
        print(f"  Min: {flat_accels.min():.4f}")
        print(f"  Max: {flat_accels.max():.4f}")
        print(f"  Std: {flat_accels.std():.4f}")
        print(f"  99th Percentile (Abs): {np.percentile(np.abs(flat_accels), 99):.4f}")

    print("\n--- Analyzing Synthetic Data ---")
    syn_files = glob.glob('data/synthetic_trajectories_*.csv')
    if syn_files:
        latest_syn = max(syn_files, key=os.path.getctime)
        print(f"Checking latest synthetic file: {latest_syn}")
        syn_df = pd.read_csv(latest_syn)
        
        # Exclude metadata
        traj_cols = [c for c in syn_df.columns if c not in ["target_len", "target_spd"]]
        syn_speeds = syn_df[traj_cols].values
        
        # Flatten and remove padding (approximate)
        # Better: just flatten non-zeros or simply flatten all (padding is 0 so it affects mean but not Max)
        syn_speeds_flat = syn_speeds.flatten()
        
        # Compute Accel (First derivative)
        # axis=1 diff
        syn_accels = np.diff(syn_speeds, axis=1).flatten()
        # Assuming format: trip_id, time_step, speed, accel... 
        # Wait, the format is columns=time_steps, rows=trips
        # Let's inspect the structure first
        print(f"Synthetic Speed:")
        print(f"  Min: {syn_speeds_flat.min():.4f}")
        print(f"  Max: {syn_speeds_flat.max():.4f}")
        print(f"  Mean: {syn_speeds_flat.mean():.4f}")
        print(f"  Std: {syn_speeds_flat.std():.4f}")
        print(f"  95th Percentile: {np.percentile(syn_speeds_flat, 95):.4f}")
        print(f"  99th Percentile: {np.percentile(syn_speeds_flat, 99):.4f}")
        
        print(f"Synthetic Accel:")
        print(f"  Min: {syn_accels.min():.4f}")
        print(f"  Max: {syn_accels.max():.4f}")
        print(f"  Std: {syn_accels.std():.4f}")
        print(f"  99th Percentile (Abs): {np.percentile(np.abs(syn_accels), 99):.4f}")

        # Distance Error Analysis
        if "target_len" in syn_df.columns and "target_spd" in syn_df.columns:
            errors = []
            for i, row in syn_df.iterrows():
                l = int(row["target_len"])
                target_v = row["target_spd"]
                # Target Distance = AvgSpeed * Duration
                target_dist = target_v * (l / 1.0) # Assuming 1Hz
                
                # Actual Distance = Sum(Speed) * dt
                # syn_speeds[i] corresponds to the i-th row speed values
                actual_speed = syn_speeds[i][:l] 
                actual_dist = np.sum(actual_speed)
                
                if target_dist > 10.0: # Avoid tiny trips div-by-zero issues
                    err = (actual_dist - target_dist) / target_dist
                    errors.append(err)
            
            if len(errors) > 0:
                errors = np.array(errors) * 100.0 # Percentage
                print(f"\nDistance Error Analysis:")
                print(f"  Mean Error (Bias): {np.mean(errors):.2f}%")
                print(f"  MAE (Accuracy):    {np.mean(np.abs(errors)):.2f}%")
                print(f"  Max Abs Error:     {np.max(np.abs(errors)):.2f}%")

if __name__ == "__main__":
    analyze_stats()
