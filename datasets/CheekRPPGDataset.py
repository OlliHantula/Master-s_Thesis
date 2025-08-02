import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.utils.rppg_utils import average_overlapping_rppg_windows
from data.utils.sequence_builder import create_sequences

# Create custom transform classes to replace Lambda transforms for multiple workers.
class ToUint8IfFloat32:
    def __call__(self, x):
        return x.astype(np.uint8) if x.dtype == np.float32 else x

class MakeContiguous:
    def __call__(self, x):
        return np.ascontiguousarray(x)

# transforms.Lambda(lambda x: x.astype(np.uint8) if x.dtype == np.float32 else x),
# transforms.Lambda(lambda x: np.ascontiguousarray(x)),
                


class CheekRPPGDataset(Dataset):
    """
    PyTorch Dataset for loading cheek patch sequences and rPPG signals
    for blood pressure prediction.

    Args:
        data_root (str): Root directory of processed data.
        split_list (List[str]): List of sample names to include.
        sequence_length (int): Number of frames per sequence.
        stride (int): Step size between sequences.
        target (str): "SBP" or "DBP".
        use_rppg (bool): Whether to include rPPG.
        rppg_mode (str): "sequence" for 1 rPPG vector per sequence,
                         "frame" for per-frame rPPG signal.
        transform (callable, optional): Optional transform to apply to image patches.
        scaler: MinMaxScale the ground truths
    """
    def __init__(
        self,
        data_root,
        split_list,
        sequence_length=200,
        stride=25,
        target="SBP",
        use_rppg=False,
        rppg_mode="sequence",
        transform=None,
        scaler=None
    ):
        self.data_root = data_root
        self.split_list = split_list
        self.sequence_length = sequence_length
        self.stride = stride
        self.target = target.upper()
        self.use_rppg = use_rppg
        self.rppg_mode = rppg_mode
        self.scaler = scaler

        # Default transform: resize to 224x224 and normalize for EfficientNetB0
        if transform is None:
            self.transform = transforms.Compose([
                # ToUint8IfFloat32(),
                MakeContiguous(),
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Build index of valid sequences
        self.samples = self._build_index()

    def _build_index(self):
        """
        Loads and slices each sample into valid sequences using create_sequences.
        Filters out sequences with NaNs in SBP or DBP.
        """
        index = []
        for sample_name in self.split_list:
            sample_dir = os.path.join(self.data_root, sample_name)
            try:
                lc = np.load(os.path.join(sample_dir, f"{sample_name}_LC.npy"))
                rc = np.load(os.path.join(sample_dir, f"{sample_name}_RC.npy"))
                sbp = np.load(os.path.join(sample_dir, f"{sample_name}_SBP.npy"))
                dbp = np.load(os.path.join(sample_dir, f"{sample_name}_DBP.npy"))
                rppg = None

                # print(f"Loading sample: {sample_name}")
                # print(f"LC shape: {lc.shape}, RC shape: {rc.shape}")
                # print(f"SBP shape: {sbp.shape}, DBP shape: {dbp.shape}")

                if self.use_rppg:
                    rppg = np.load(os.path.join(sample_dir, f"{sample_name}_rPPG.npy"))
                    # print(f"raw rPPG shape: {rppg.shape}")
                    if self.rppg_mode == "frame":
                        # print("Calculating avg_rppg...")
                        rppg = average_overlapping_rppg_windows(
                            rppg_windows=rppg,
                            stride=self.stride
                        ) 
                    # print(f"rPPG shape: {rppg.shape}")


                # Slice sequences and filter out invalid ones
                lc_seqs, rc_seqs, sbp_seqs, dbp_seqs, rppg_seqs = create_sequences(
                    lc, rc, sbp, dbp, rppg,
                    sequence_length=self.sequence_length,
                    stride=self.stride,
                    include_rppg=self.use_rppg,
                    rppg_mode=self.rppg_mode
                )
                # print(f"Generated {len(lc_seqs)} sequences for {sample_name}")
                # print(f"lc: {lc_seqs.shape}, rc: {rc_seqs.shape}, sbp: {sbp_seqs.shape}, dbp: {dbp_seqs.shape}, rppg: {rppg_seqs.shape}")

                for i in range(len(sbp_seqs)):
                    index.append({
                        "lc": lc_seqs[i],
                        "rc": rc_seqs[i],
                        "label": sbp_seqs[i] if self.target == "SBP" else dbp_seqs[i],
                        "rppg": rppg_seqs[i] if self.use_rppg else None
                    })

            except Exception as e:
                print(f"Skipping {sample_name} due to error: {e}")
        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single sample with transformed cheek patches and optional rPPG.
        """
        entry = self.samples[idx]
        lc_seq = np.stack([self.transform(img) for img in entry["lc"]])
        rc_seq = np.stack([self.transform(img) for img in entry["rc"]])

        label_value = entry["label"]
        if self.scaler is not None:
            label_value = self.scaler.transform([[label_value]])[0][0]
        label = torch.tensor(label_value, dtype=torch.float32)

        sample = {
            "left_cheek": torch.tensor(lc_seq),  # (T, C, H, W)
            "right_cheek": torch.tensor(rc_seq),
            "label": label
        }

        if self.use_rppg:
            rppg_seq = entry["rppg"]
            sample["rppg"] = torch.tensor(rppg_seq)

        return sample

