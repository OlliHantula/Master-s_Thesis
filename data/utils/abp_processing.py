import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from scipy.interpolate import interp1d

def design_lowpass_filter(fs=1000, highcut=20, order=6):
    """Design a lowpass Butterworth filter."""
    return butter(order, highcut, btype='low', fs=fs, output='sos')

def apply_filter(signal, sos):
    """Apply a second-order sections (SOS) filter to the signal."""
    return sosfilt(sos, signal)

def interpolate_with_nan(peaks, signal, frame_indices, threshold=3, min_value=10, max_value=250):
    """Interpolate signal at frame indices, masking outliers before and after a bad peak as NaN."""
    values = signal[peaks]
    mean = np.nanmean(values)
    std = np.nanstd(values)

    bad_indices = np.where(
        (values < mean - threshold * std) |
        (values > mean + threshold * std) |
        (values < min_value) |
        (values > max_value)
    )[0]

    valid_mask = np.ones_like(frame_indices, dtype=bool)
    for idx in bad_indices:
        if 0 < idx < len(peaks) - 1:
            start = peaks[idx - 1]
            end = peaks[idx + 1]
            mask = (frame_indices >= start) & (frame_indices <= end)
            valid_mask[mask] = False
        elif idx == 0:
            mask = frame_indices <= peaks[1]
            valid_mask[mask] = False
        elif idx == len(peaks) - 1:
            mask = frame_indices >= peaks[-2]
            valid_mask[mask] = False

    interp_func = interp1d(peaks, signal[peaks], kind='linear',
                           fill_value=(signal[peaks[0]], signal[peaks[-1]]),
                           bounds_error=False)
    interpolated = interp_func(frame_indices)
    interpolated[~valid_mask] = np.nan
    return interpolated

def extract_sbp_dbp(abp_signal, max_hr, fs=1000, fps=25):
    """Extract SBP and DBP values from ABP signal."""
    sos = design_lowpass_filter(fs=fs)
    filtered_abp = apply_filter(abp_signal, sos)

    dist = int(fs / (max_hr / 60) * 1.05)  # Convert max HR to peak distance * some margin
    dbp_peaks, _ = find_peaks(-filtered_abp, distance=dist)

    sbp_peaks = []
    dbp_peaks_filtered = []

    for i in range(len(dbp_peaks) - 1):
        segment = filtered_abp[dbp_peaks[i]:dbp_peaks[i + 1]]
        sbp_idx = np.argmax(segment) + dbp_peaks[i]
        dbp_idx = dbp_peaks[i]

        # Check if SBP and DBP are too close in value = pulse pressure < 5
        if abs(filtered_abp[sbp_idx] - filtered_abp[dbp_idx]) >= 5:
            sbp_peaks.append(sbp_idx)
            dbp_peaks_filtered.append(dbp_idx)

    # Remove first peak if it's an outlier
    if len(dbp_peaks_filtered) > 1 and abs(filtered_abp[dbp_peaks_filtered[1]] - filtered_abp[dbp_peaks_filtered[0]]) > 10:
        dbp_peaks_filtered = dbp_peaks_filtered[1:]
    if len(sbp_peaks) > 1 and abs(filtered_abp[sbp_peaks[1]] - filtered_abp[sbp_peaks[0]]) > 10:
        sbp_peaks = sbp_peaks[1:]

    frame_indices = np.linspace(0, len(abp_signal) - 1, int(len(abp_signal) / (fs / fps)))
    sbp_interp = interpolate_with_nan(sbp_peaks, filtered_abp, frame_indices)
    dbp_interp = interpolate_with_nan(dbp_peaks_filtered, filtered_abp, frame_indices)

    return sbp_interp, dbp_interp, sbp_peaks, dbp_peaks_filtered
