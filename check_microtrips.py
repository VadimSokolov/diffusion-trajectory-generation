
import os
import glob

# Try relative to reproduce/
path = "../../data/Microtrips"

try:
    if os.path.exists(path):
        print(f"Path {path} exists!")
        files = glob.glob(os.path.join(path, "results_trip_*.csv"))
        print(f"Found {len(files)} files.")
        if len(files) > 0:
            print("First 5 files:", files[:5])
    else:
        print(f"Path {path} does NOT exist.")
        # Try ../data
        path2 = "../data/Microtrips"
        if os.path.exists(path2):
            print(f"Path {path2} exists!")
            files = glob.glob(os.path.join(path2, "results_trip_*.csv"))
            print(f"Found {len(files)} files.")
        else:
             print(f"Path {path2} does NOT exist.")

except Exception as e:
    print(f"Error: {e}")
