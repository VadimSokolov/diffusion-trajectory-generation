
def load_real_distribution_data(data_path):
    """Load all real trip data for distribution plotting"""
    print(f"Loading real data from {data_path}...")
    # Matches evaluate_csdi.py logic
    files = glob.glob(os.path.join(data_path, "results_trip_*.csv"))
    
    speed_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "speedMps" in df.columns:
                speed_list.append(df["speedMps"].values)
            elif "speed" in df.columns: # fallback
                 speed_list.append(df["speed"].values)
        except:
            pass
            
    if len(speed_list) > 0:
        return np.concatenate(speed_list)
    else:
        print("Warning: No real data found for distributions, using fallback")
        return np.random.normal(17, 4.5, 10000)
