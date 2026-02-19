
import pandas as pd
import numpy as np
import time

def analyze_behavior():
    try:
        # Load features
        df = pd.read_csv("data/processed/behavior_features.csv")
        
        # Assume "Friend" data is the most recent data (e.g., last 50 samples if collected recently)
        # Or split by timestamp if we know when the test started.
        # For now, let's look at the distribution of the entire dataset vs the last 5 minutes.
        
        current_time = time.time()
        # Filter for data in last 10 minutes (approx 600 seconds)
        recent_data = df[df['timestamp'] > (current_time - 600)]
        
        # Historical data (everything else)
        historical_data = df[df['timestamp'] <= (current_time - 600)]
        
        if len(recent_data) == 0:
            print("No recent data found for comparison. Using last 100 samples vs rest.")
            recent_data = df.tail(100)
            historical_data = df.iloc[:-100]
            
        print(f"Comparing {len(historical_data)} Owner samples vs {len(recent_data)} Recent/Friend samples")
        
        # Features to compare (based on likely available columns)
        # We need to know exact column names, but standard ones usually include:
        # cursor_velocity, cursor_acceleration, typing_speed, keystroke_latency
        
        features_to_check = [col for col in df.columns if col not in ['timestamp', 'label', 'window_size']]
        
        print("\n--- Behavior Comparison ---")
        print(f"{'Feature':<30} | {'Owner Avg':<15} | {'Friend/Recent Avg':<15} | {'Diff (%)':<10}")
        print("-" * 80)
        
        for feature in features_to_check:
            owner_avg = historical_data[feature].mean()
            friend_avg = recent_data[feature].mean()
            
            if owner_avg != 0:
                diff_pct = ((friend_avg - owner_avg) / abs(owner_avg)) * 100
            else:
                diff_pct = 0.0
                
            print(f"{feature:<30} | {owner_avg:.4f}          | {friend_avg:.4f}           | {diff_pct:+.1f}%")

    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    analyze_behavior()
