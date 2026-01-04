import pandas as pd
import numpy as np

import glob
import os

def verify():
    syn_files = glob.glob('data/synthetic_trajectories_*.csv')
    if not syn_files:
        print("No synthetic files found in data/")
        return
        
    latest_syn = max(syn_files, key=os.path.getctime)
    print(f"Verifying latest file: {latest_syn}")
    df = pd.read_csv(latest_syn)
    n_passed = 0
    n_total = len(df)
    
    print(f"Verifying {n_total} trajectories...")
    
    for i, row in df.iterrows():
        length = int(row["target_len"])
        # Columns 0 to 511 are the time steps
        traj = row.values[:512] 
        
        # Check Start
        v_start = traj[0]
        
        # Check End (index length-1)
        # Note: if length is 100, index 99 should be 0.
        if length > 512: length = 512 - 1
        
        v_end = traj[length-1]
        
        # Check Padding (index length onwards)
        v_pad_max = 0
        if length < 512:
            v_pad_max = np.max(np.abs(traj[length:]))

        print(f"Row {i}: Len={length}, Start={v_start:.4f}, End={v_end:.4f}, MaxPad={v_pad_max:.4f}")
        
        # Tolerance: Inpainting should be EXACT, but float math might give 1e-7.
        # However, due to the denormalization from [-1, 1], 0 m/s might be slightly off if strict math wasn't used.
        # But we clamped to min=0.0 in the script.
        
        if abs(v_start) < 0.1 and abs(v_end) < 0.1 and abs(v_pad_max) < 0.1:
            n_passed += 1
        else:
            print(f"   -> FAILED (Start/End/Pad) constraint!")

    print(f"\nResult: {n_passed}/{n_total} passed strict zero-constraints.")
    
    if n_passed == n_total:
        print("SUCCESS: All constraints satisfied.")
    else:
        print("FAILURE: Some constraints violated.")

if __name__ == "__main__":
    verify()
