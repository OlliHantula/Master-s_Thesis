import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from datasets.CheekRPPGDataset import CheekRPPGDataset
# Set environment variable to avoid KMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set data paths
train_data_root = "./data/processed/train/"
val_data_root = "./data/processed/val/"
test_data_root = "./data/processed/test/"
train_subjects = sorted([d for d in os.listdir(train_data_root) if os.path.isdir(os.path.join(train_data_root, d))])
val_subjects = sorted([d for d in os.listdir(val_data_root) if os.path.isdir(os.path.join(val_data_root, d))])
test_subjects = sorted([d for d in os.listdir(test_data_root) if os.path.isdir(os.path.join(test_data_root, d))])

data_roots = {
    "Train": train_data_root,
    "Validation": val_data_root,
    "Test": test_data_root,
}
subject_lists = {
    "Train": train_subjects,
    "Validation": val_subjects,
    "Test": test_subjects
}

# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
targets = ["SBP", "DBP"]
sequence_lengths = [25, 200]

for i, target in enumerate(targets):
    for j, sequence_length in enumerate(sequence_lengths):
        ax = axes[j][i]
        for split in ["Train", "Validation", "Test"]:
            dataset = CheekRPPGDataset(
                data_root=data_roots[split],
                split_list=subject_lists[split],
                sequence_length=sequence_length,
                stride=sequence_length,
                target=target
            )
            labels = np.array([entry["label"] for entry in dataset.samples]).reshape(-1, 1)
            np.save(f"{split}_{target}_seq{sequence_length}_labels.npy", labels)
            ax.hist(labels, bins=np.arange(labels.min(), labels.max() + 1), alpha=0.5, label=split)

        ax.set_title(f"{target} Distribution (Sequence Lenght: {sequence_length})")
        ax.set_xlabel(f"{target} mmHg")
        ax.set_ylabel("Frequency")
        # Add subplot label
        label_index = i + j * 2
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(-0.1, 1.1, subplot_labels[label_index], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.savefig("Combined_Distributions.png", dpi=300)
