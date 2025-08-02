import numpy as np

def create_sequences(
    lc_imgs,
    rc_imgs,
    sbp,
    dbp,
    rppg=None,
    sequence_length=200,
    stride=25,
    include_rppg=True,
    rppg_mode="sequence"
):
    """
    Slice full-length sequences into fixed-length windows.
    Filters out sequences with NaNs or inconsistent shapes.

    Args:
        lc_imgs (np.ndarray): Left cheek image sequence (T, H, W, C)
        rc_imgs (np.ndarray): Right cheek image sequence (T, H, W, C)
        sbp (np.ndarray): SBP values per frame (T,)
        dbp (np.ndarray): DBP values per frame (T,)
        rppg (np.ndarray): rPPG signal (T,) or (N, T) if windowed
        sequence_length (int): Number of frames per sequence
        stride (int): Step size between sequences
        include_rppg (bool): Whether to include rPPG in output
        rppg_mode (str): "sequence" or "frame"

    Returns:
        Tuple of np.ndarrays: (lc_seqs, rc_seqs, sbp_seqs, dbp_seqs, rppg_seqs)
    """
    num_frames = len(lc_imgs)
    lc_seqs, rc_seqs, sbp_seqs, dbp_seqs, rppg_seqs = [], [], [], [], []

    for start in range(0, num_frames - sequence_length + 1, stride):
        end = start + sequence_length

        sbp_window = sbp[start:end]
        dbp_window = dbp[start:end]

        if np.isnan(sbp_window).any() or np.isnan(dbp_window).any():
            continue

        lc_window = lc_imgs[start:end]
        rc_window = rc_imgs[start:end]

        # Check image shape consistency
        if not all(img.shape == lc_window[0].shape for img in lc_window):
            print(f"Skipping left cheek window due to inconsistent shapes at frames {start}-{end}")
            continue
        if not all(img.shape == rc_window[0].shape for img in rc_window):
            print(f"Skipping right cheek window due to inconsistent shapes at frames {start}-{end}")
            continue

        # Handle rPPG
        rppg_window = None
        if include_rppg and rppg is not None:
            if rppg_mode == "sequence":
                try:
                    rppg_window = rppg[start // stride].flatten()
                except IndexError:
                    print(f"Skipping rPPG sequence window at index {start // stride} due to out-of-bounds")
                    continue
            elif rppg_mode == "frame":
                rppg_window = rppg[start:end]
            if rppg_window.shape[0] != sequence_length:
                print(f"Skipping rPPG frame window due to inconsistent length at frames {start}-{end}")
                continue

        lc_seqs.append(lc_window)
        rc_seqs.append(rc_window)
        sbp_seqs.append(np.mean(sbp_window))
        dbp_seqs.append(np.mean(dbp_window))
        if include_rppg and rppg_window is not None:
            rppg_seqs.append(rppg_window)

    return (
        np.array(lc_seqs),
        np.array(rc_seqs),
        np.array(sbp_seqs),
        np.array(dbp_seqs),
        np.array(rppg_seqs) if include_rppg and rppg_seqs else None
    )
