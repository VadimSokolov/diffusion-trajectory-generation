import os
import subprocess
import argparse

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=500, help="Number of real files to train on")
    parser.add_argument('--data_path', type=str, default="/Users/vsokolov/Dropbox/prj/svtrip/data/Microtrips")
    args = parser.parse_args()

    # 1. Train
    print("--- Step 1: Training ---")
    limit_arg = f"--limit_files {args.sample_size}" if args.sample_size > 0 else ""
    run_command(f"python -u diffusion_trajectory.py --train --epochs {args.epochs} {limit_arg} --data_path '{args.data_path}'")
    
    # 2. Generate
    print("--- Step 2: Generation ---")
    # Generating 100 samples for evaluation
    run_command("python diffusion_trajectory.py --generate")
    
    # 3. Evaluate
    print("--- Step 3: Evaluation ---")
    run_command("python evaluate_distribution.py")
    
    print("--- Loop Complete ---")
    print("Check 'report/' for evaluation plots and reports.")

if __name__ == "__main__":
    main()
