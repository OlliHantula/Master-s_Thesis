from pyVHR.BVP import RGB_sig_to_BVP, BPfilter, cupy_CHROM, cpu_CHROM, rgb_filter_th, apply_filter
from pyVHR.extraction import SignalProcessing, SkinExtractionConvexHull, sig_windowing

def calculate_rppg(video_path, fps=25, wsize=8, stride=1, use_gpu=True):
    """
    Extract rPPG signal from a video using pyVHR.
    
    Args:
        video_path (str): Path to the video file.
        fps (int): Frames per second of the video.
        wsize (int): Window size in seconds.
        stride (int): Stride in seconds.
        use_gpu (bool): Whether to use GPU acceleration.

    Returns:
        np.ndarray: Filtered BVP signal windows.
    """
    sig_extractor = SignalProcessing()

    if use_gpu:
        sig_extractor.display_cuda_device()
        sig_extractor.choose_cuda_device(0)
        sig_extractor.set_skin_extractor(SkinExtractionConvexHull('GPU'))

    # Set landmark indices (100 equispaced landmarks)
    landmarks = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103,
                 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188,
                 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284,
                 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411,
                 412, 417, 421, 425, 426, 427, 430, 432, 436]
    sig_extractor.set_landmarks(landmarks)

    # Extract holistic RGB signal
    hol_sig = sig_extractor.extract_holistic(video_path)

    # Window the signal
    windowed_sig, _ = sig_windowing(hol_sig, wsize, stride, fps)

    # Apply RGB thresholding
    filtered_rgb = apply_filter(windowed_sig, rgb_filter_th, fps=fps,
                                        params={'RGB_LOW_TH': 0, 'RGB_HIGH_TH': 240})

    # Apply bandpass filter
    filtered_rgb = apply_filter(filtered_rgb, BPfilter,
                                        params={'order': 6, 'minHz': 0.65, 'maxHz': 4.0, 'fps': fps})

    # Extract BVP using CHROM method
    method = cupy_CHROM if use_gpu else cpu_CHROM
    device = 'cuda' if use_gpu else 'cpu'
    bvp = RGB_sig_to_BVP(filtered_rgb, fps, device_type=device, method=method)

    # Final bandpass filter
    bvp = apply_filter(bvp, BPfilter,
                        params={'order': 6, 'minHz': 0.65, 'maxHz': 4.0, 'fps': fps})

    return bvp
