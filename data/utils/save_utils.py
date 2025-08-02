import os
import numpy as np

def save_full_sample(
    sample_dir,
    sample_name,
    left_cheek,
    right_cheek,
    sbp,
    dbp,
    rppg=None
):
    """
    Save full-length sequences to disk.

    Args:
        sample_dir (str): Directory to save the sample.
        sample_name (str): Base name for saved files.
        left_cheek (np.ndarray): Full left cheek sequence.
        right_cheek (np.ndarray): Full right cheek sequence.
        sbp (np.ndarray): Full SBP signal.
        dbp (np.ndarray): Full DBP signal.
        rppg (np.ndarray, optional): Windowed rPPG signal.
    """
    os.makedirs(sample_dir, exist_ok=True)

    np.save(os.path.join(sample_dir, f"{sample_name}_LC.npy"), left_cheek)
    np.save(os.path.join(sample_dir, f"{sample_name}_RC.npy"), right_cheek)
    np.save(os.path.join(sample_dir, f"{sample_name}_SBP.npy"), sbp)
    np.save(os.path.join(sample_dir, f"{sample_name}_DBP.npy"), dbp)

    if rppg is not None:
        np.save(os.path.join(sample_dir, f"{sample_name}_rPPG.npy"), rppg)
