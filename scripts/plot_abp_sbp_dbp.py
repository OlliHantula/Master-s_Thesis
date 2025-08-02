
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom function
from data.utils.abp_processing import extract_sbp_dbp

# Constants
FS = 1000  # ABP sampling rate
FPS = 25   # Video frame rate

# Base path to V4V dataset
BASE_PATH = "./data/V4V_dataset"

# Output folder for plots
PLOT_DIR = "abp_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Dataset configurations
DATASETS = {
    "train": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Ground truth/BP_raw_1KHz"),
        "hr_dir": os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Ground truth/Physiology"),
        "video_dirs": [
            os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/train-001_of_002"),
            os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/train-002_of_002")
        ],
        "hr_type": "matrix"
    },
    "val": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 2_ Testing set/blood_pressure/val_set_bp"),
        "hr_file": os.path.join(BASE_PATH, "Phase 2_ Testing set/validation_set_gt_release.txt"),
        "video_dir": os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/valid"),
        "hr_type": "dict"
    },
    "test": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 2_ Testing set/blood_pressure/test_set_bp"),
        "hr_file": os.path.join(BASE_PATH, "Phase 2_ Testing set/test_set_gt_release.txt"),
        "video_dir": os.path.join(BASE_PATH, "Phase 2_ Testing set/Videos/Test/test"),
        "hr_type": "dict"
    }
}

# Load HR dictionary from text file
def load_hr_dict(hr_file):
    hr_dict = {}
    with open(hr_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            sample = parts[0]
            data_type = parts[1]
            if data_type == 'HR':
                hr_values = list(map(float, parts[2:]))
                hr_dict[sample] = hr_values
    return hr_dict

# Plot ABP with SBP and DBP
def plot_abp_sbp_dbp(abp, sbp, dbp, sample_name):
    # Create time axes
    t_abp = np.arange(len(abp)) / FS
    t_sbp_dbp = np.linspace(0, len(abp) / FS, len(sbp))

    plt.figure(figsize=(10, 6))
    plt.plot(t_abp, abp, label='ABP', color='blue')
    plt.plot(t_sbp_dbp, sbp, label='SBP', color='red', linewidth=2)
    plt.plot(t_sbp_dbp, dbp, label='DBP', color='green', linewidth=2)
    plt.title(f"ABP with SBP/DBP - {sample_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{sample_name}.png"))
    plt.close()

# Process each dataset
for split, config in DATASETS.items():
    print(f"Processing {split} set...")

    if config["hr_type"] == "matrix":
        abp_dir = config["abp_dir"]
        hr_dir = config["hr_dir"]
        for video_dir in config["video_dirs"]:
            for file in os.listdir(video_dir):
                if not file.endswith(".mkv"):
                    continue
                sample_name = file.replace(".mkv", "")
                abp_path = os.path.join(abp_dir, sample_name.replace("_", "-") + "-BP.txt")
                hr_path = os.path.join(hr_dir, sample_name + ".txt")
                if not os.path.exists(abp_path) or not os.path.exists(hr_path):
                    continue
                try:
                    abp = np.loadtxt(abp_path)
                    # Open the HR file
                    with open(hr_path, 'r') as hr_file:
                        hr_line = hr_file.readline().strip().split(', ')
                        hr = list(map(float, hr_line[2:]))
                        max_hr = np.max(hr)
                    sbp, dbp = extract_sbp_dbp(abp, max_hr, fs=FS, fps=FPS)
                    plot_abp_sbp_dbp(abp, sbp, dbp, sample_name)
                except Exception as e:
                    print(f"Failed to process {sample_name}: {e}")

    elif config["hr_type"] == "dict":
        abp_dir = config["abp_dir"]
        video_dir = config["video_dir"]
        hr_dict = load_hr_dict(config["hr_file"])
        for file in os.listdir(video_dir):
            if not file.endswith(".mkv"):
                continue
            sample_name = file.replace(".mkv", "")
            abp_path = os.path.join(abp_dir, sample_name.replace("_", "-") + ".txt")
            if not os.path.exists(abp_path) or file not in hr_dict:
                print(f"Skipping {sample_name}: Missing ABP or HR data")
                continue
            try:
                abp = np.loadtxt(abp_path)
                max_hr = np.max(hr_dict[file])
                sbp, dbp = extract_sbp_dbp(abp, max_hr, fs=FS, fps=FPS)
                plot_abp_sbp_dbp(abp, sbp, dbp, sample_name)
            except Exception as e:
                print(f"Failed to process {sample_name}: {e}")
