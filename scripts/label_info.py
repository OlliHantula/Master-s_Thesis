import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

# Configuration options
targets = ["SBP", "DBP"]
sequence_lengths = [25, 200]
splits = ["Train", "Validation", "Test"]

# Dictionary to store statistics
label_stats = {}

# Iterate through all configurations
for target in targets:
    for seq_len in sequence_lengths:
        for split in splits:
            filepath = f"{split}_{target}_seq{seq_len}_labels.npy"
            if os.path.exists(filepath):
                labels = np.load(filepath)
                mean_val = np.mean(labels)
                std_val = np.std(labels)
                count = len(labels)
                label_stats[(target, seq_len, split)] = {
                    "mean": mean_val,
                    "std": std_val,
                    "count": count
                }
                print(f"{target} | SeqLen: {seq_len} | {split}:")
                print(f"  Count: {count}")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}\n")
            else:
                print(f"File not found: {filepath}")

