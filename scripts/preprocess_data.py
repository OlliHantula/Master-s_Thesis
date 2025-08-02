import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv
import cv2
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.utils.abp_processing import extract_sbp_dbp
from data.utils.roi_processing import process_video
from data.utils.rppg_extraction import calculate_rppg
from data.utils.save_utils import save_full_sample

# Constants
FS = 1000  # ABP sampling rate
FPS = 25   # Video frame rate
MIN_VIDEO_DURATION_SEC = 9
MIN_VIDEO_FRAMES = MIN_VIDEO_DURATION_SEC * FPS

# Dataset paths
BASE_PATH = "./data/V4V_dataset"
DATASETS = {
    "train": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Ground truth/BP_raw_1KHz"),
        "hr_dir": os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Ground truth/Physiology"),
        "video_dirs": [
            os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/train-001_of_002"),
            os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/train-002_of_002")
        ],
        "output_dir": "./data/processed/train"
    },
    "val": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 2_ Testing set/blood_pressure/val_set_bp"),
        "hr_file": os.path.join(BASE_PATH, "Phase 2_ Testing set/validation_set_gt_release.txt"),
        "video_dirs": [os.path.join(BASE_PATH, "Phase 1_ Training_Validation sets/Videos/valid")],
        "output_dir": "./data/processed/val"
    },
    "test": {
        "abp_dir": os.path.join(BASE_PATH, "Phase 2_ Testing set/blood_pressure/test_set_bp"),
        "hr_file": os.path.join(BASE_PATH, "Phase 2_ Testing set/test_set_gt_release.txt"),
        "video_dirs": [os.path.join(BASE_PATH, "Phase 2_ Testing set/Videos/Test/test")],
        "output_dir": "./data/processed/test"
    }
}

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

def process_dataset(split_name, config, skipped_samples):
    if split_name in ["val", "test"]:
        print("Loading HR dict...")
        hr_dict = load_hr_dict(config["hr_file"])
        print("HR dict loaded with", len(hr_dict), "entries.")

    for video_dir in config["video_dirs"]:
        print(f"Processing video_dir: {video_dir}")
        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mkv")]

        for video_file in video_files:
            sample_name = video_file.replace(".mkv", "")
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing sample: {sample_name}")

            # Check video length
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count < MIN_VIDEO_FRAMES:
                    skipped_samples.append((split_name, sample_name, "Too short"))
                    continue
            except Exception as e:
                skipped_samples.append((split_name, sample_name, f"Video error: {e}"))
                continue
            
            # Create sample directory
            sample_dir = os.path.join(config["output_dir"], sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            if split_name == "train":
                abp_path = os.path.join(config["abp_dir"], sample_name.replace("_", "-") + "-BP.txt")
                hr_path = os.path.join(config["hr_dir"], sample_name + ".txt")
                if not os.path.exists(abp_path) or not os.path.exists(hr_path):
                    skipped_samples.append((split_name, sample_name, "Missing ABP or HR"))
                    continue
                try:
                    abp = np.loadtxt(abp_path)
                    # Open the HR file
                    with open(hr_path, 'r') as hr_file:
                        hr_line = hr_file.readline().strip().split(', ')
                        hr = list(map(float, hr_line[2:]))
                        max_hr = np.max(hr)
                except Exception as e:
                    print(f"Failed to process {sample_name}: {e}")

            else:
                abp_path = os.path.join(config["abp_dir"], sample_name.replace("_", "-") + ".txt")
                if not os.path.exists(abp_path) or video_file not in hr_dict:
                    skipped_samples.append((split_name, sample_name, "Missing ABP or HR"))
                    continue
                abp = np.loadtxt(abp_path)
                max_hr = np.max(hr_dict[video_file])

            sbp, dbp = extract_sbp_dbp(abp, max_hr, fs=FS, fps=FPS)
            left_cheek, right_cheek = process_video(video_path)
            rppg_windows = calculate_rppg(video_path, fps=FPS, wsize=8, stride=1, use_gpu=True)

            save_full_sample(
                sample_dir=sample_dir,
                sample_name=sample_name,
                left_cheek=left_cheek,
                right_cheek=right_cheek,
                sbp=sbp,
                dbp=dbp,
                rppg=rppg_windows
            )

            print(f"[{split_name}] Processed and saved: {sample_name}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess rPPG dataset")
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all",
        help="Which dataset split to preprocess"
    )
    args = parser.parse_args()

    skipped_samples = []

    if args.split == "all":
        for split_name, config in DATASETS.items():
            print(f"\nProcessing {split_name} dataset...")
            process_dataset(split_name, config, skipped_samples)
    else:
        print(f"\nProcessing {args.split} dataset...")
        process_dataset(args.split, DATASETS[args.split], skipped_samples)

    # Save skipped samples
    if skipped_samples:
        with open("skipped_samples.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Split", "Sample", "Reason"])
            writer.writerows(skipped_samples)
        print(f"\nSkipped samples saved to skipped_samples.csv")

if __name__ == "__main__":
    main()

