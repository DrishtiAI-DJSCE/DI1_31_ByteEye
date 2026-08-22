import pandas as pd
import numpy as np
import os

if not os.path.exists("data/diagnostics.csv"):
    print("No diagnostics.csv found.")
    exit(0)
    
df = pd.read_csv("data/diagnostics.csv")

print("=== DIAGNOSTIC DATA ANALYSIS ===")
print(f"Total Rows: {len(df)}")
print(f"Unique Tracks: {df['track_id'].nunique()}")
print("---------------------------------")

df['yaw_deviation'] = pd.to_numeric(df['yaw_deviation'], errors='coerce')
df['width_ratio'] = pd.to_numeric(df['shoulder_width_ratio'], errors='coerce')

normal = df[df['candidate_anomaly'] == 'NORMAL']
candidate = df[df['candidate_anomaly'] == 'CANDIDATE']

print("NORMAL BEHAVIOUR DISTRIBUTIONS:")
print("Yaw Deviation (Nose/Shoulder):")
if not normal.empty and not normal['yaw_deviation'].isna().all():
    print(f"  Mean: {normal['yaw_deviation'].mean():.3f}")
    print(f"  95th Percentile: {normal['yaw_deviation'].quantile(0.95):.3f}")
    print(f"  99th Percentile: {normal['yaw_deviation'].quantile(0.99):.3f}")
else:
    print("  No valid data.")
    
print("\nShoulder Width Ratio (vs Baseline):")
if not normal.empty and not normal['width_ratio'].isna().all():
    print(f"  Mean: {normal['width_ratio'].mean():.3f}")
    print(f"  5th Percentile: {normal['width_ratio'].quantile(0.05):.3f}")
    print(f"  1st Percentile: {normal['width_ratio'].quantile(0.01):.3f}")
else:
    print("  No valid data.")

print("---------------------------------")
print("CANDIDATE ANOMALY DISTRIBUTIONS (Current triggers):")
print("Yaw Deviation:")
if not candidate.empty and not candidate['yaw_deviation'].isna().all():
    print(f"  Mean: {candidate['yaw_deviation'].mean():.3f}")
    print(f"  Min: {candidate['yaw_deviation'].min():.3f}")
else:
    print("  No valid data.")

print("\nShoulder Width Ratio:")
if not candidate.empty and not candidate['width_ratio'].isna().all():
    print(f"  Mean: {candidate['width_ratio'].mean():.3f}")
    print(f"  Max: {candidate['width_ratio'].max():.3f}")
else:
    print("  No valid data.")
