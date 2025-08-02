import numpy as np

def average_overlapping_rppg_windows(rppg_windows, stride=25):
    """
    Convert overlapping rPPG windows into a per-frame averaged signal.

    Args:
        rppg_windows (np.ndarray): Shape (N, 1, W), where N is number of windows, W is window size
        window_size (int): Number of frames per window
        stride (int): Stride between windows

    Returns:
        np.ndarray: Averaged rPPG signal per frame (T,)
    """
    window_size = rppg_windows.shape[2]
    T = (rppg_windows.shape[0] - 1) * stride + window_size
    rppg_sum = np.zeros(T)
    rppg_count = np.zeros(T)

    for i, window in enumerate(rppg_windows):
        window = window.flatten() # Convert (1,200) to (200,)
        start = i * stride
        end = start + window_size
        rppg_sum[start:end] += window
        rppg_count[start:end] += 1

    rppg_avg = np.divide(rppg_sum, rppg_count, out=np.zeros_like(rppg_sum), where=rppg_count != 0)
    return rppg_avg
