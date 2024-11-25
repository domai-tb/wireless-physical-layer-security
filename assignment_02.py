import numpy as np
import matplotlib.pyplot as plt

# usefull function for generating a windowed version of an array
from numpy.lib.stride_tricks import sliding_window_view

# library for dwt with python
import pywt

"""
This File contains all functions that you have to implement. Please read the docstrings of each function.
Please rename the last to feature names in line 29 and 30, after you have implemented you own features.
If you have problems implementing the function ask us in the exercise session or write us via moodle.
"""


FEATURE_NAMES = [
    "Normalized Amplitude",
    "Normalized Phase",
    "Normalized Frequency",
    "Change in Amplitude",
    "Normalized In-Phase",
    "Normalized Quadrature",
    "Mean-Centered Normalized Amplitude",
    "Power per Section",
    "Mean Centered Normalized Phase",
    "Change in DWT Coeff.",
    "Length of Transient",
    "Difference normalized mean and normalized maximum",
]


def _extract_transients_smartphone(
    data: np.ndarray,
) -> np.ndarray:
    """Function that implements the transient detection. The goal is to
    fill the _transients array with transient start- and end-points. _transients[i, 0] is the start-
    point of the i-th signal and _transients[i, 1] is its endpoint.

    Args:
        data (np.ndarray): The Raw IQ-Data from one smartphone shape=(#signals, #Samples_per_signal)

    Returns:
        np.ndarray: The extracted transients with shape (#signals, 2).
    """
    k       = 1     # ggf. auf 0.85?; orig = 1
    w       = 100   # orig = 100
    alpha   = 0.1   # orig = 0.1; opt = 0.25
    th      = 1e-1  # 0.05
    clws    = 1000  # original 1000 Signal 22 -> erstes 0 Element bei 5857  -> 4945  4955

    # We use the amplitude of the signals for the transient detection
    signal = np.abs(data)
    n_signals, n_samples_per_signal = data.shape
    _transients = np.zeros((n_signals, 2))
    ########## Your code starts here ##########
    
    for i in range(n_signals):
        amplitude = signal[i]  # Amplitude of the signal
        sliding_windows = sliding_window_view(amplitude, window_shape=w, axis=-1)
        variances = k * np.var(sliding_windows, axis=-1, ddof=1)
        
        # Initialize CUSUM
        cusum = np.zeros_like(variances)
        #detection_signal = np.zeros_like(variances)
        detection_signal = [0 for _ in  range(w)]
        # Calculate the detection signal using CUSUM
        for t in range(1, len(variances)):
            delta = variances[t] - variances[t - 1]
            cusum[t] = max(cusum[t - 1] + delta - alpha, 0)
            #detection_signal[t] = cusum[t]
            detection_signal.append(cusum[t])

        detection_signal = np.array(detection_signal)
        # Detect the start of the transient
        end_idx = -1
        start_idx = -1
        try:
            poss_start_idx = np.nonzero(detection_signal > th)[0] # point above threshold
            sliding_windows = sliding_window_view(detection_signal, window_shape=clws)

            count = 1
            for window in sliding_windows:
                if np.argmax(window) == 0 and np.max(window) > 0:
                    end_idx = count
                    break
                count += 1

            poss_start_idx = poss_start_idx[np.where(poss_start_idx < end_idx)[0]]
            for idx in reversed(poss_start_idx):
                if detection_signal[idx-1] < th:
                   start_idx = idx
                   break
        except IndexError:
            # Default values if no transient is found
            start_idx, end_idx = 0, 0

        if start_idx == end_idx:
            index = np.argmin(detection_signal[start_idx: detection_signal.size - 1])
            min_val = np.min(detection_signal[start_idx: detection_signal.size - 1])
            print("Transienten Problem bei Signal: " + str(i))
            end_idx = start_idx + 150
        # Store results
        _transients[i, 0] = start_idx
        _transients[i, 1] = end_idx

        # DEBUG PLOTING
        #plt.plot(signal[i], label="signal")
        #plt.plot(variances, label="variances")
        #plt.plot(detection_signal, label="detection_signal")
        #plt.plot(amplitude, label="amplitude")
        #plt.plot([start_idx], [10], '|', label=f"Start Transient: {start_idx}")
        #plt.plot([end_idx], [10], '|', label=f"End Transient: {end_idx}")
        #plt.plot(poss_start_idx, [0]*len(poss_start_idx), "r.", label="Mögliche Start Punkte")
        #plt.legend(loc="upper left")
        #plt.show()
        #exit()
            

    ########## Your code ends here ##########
    return _transients


def _extract_features_smartphone(
    data: np.ndarray,
    transients: np.ndarray,
) -> np.ndarray:
    """This function extracts the features from the raw measurements of a single smartphone. It's important
    to only use the transient part of the signal. Therefore the transients of each signal are given as well.
    The goal is to fill the _features array. In the end _features[i, j] is the j-th feature of the i-th signal
    of the smartphone. Its important to have the same order of features as given in FEATURE_NAMES. So _features[i, 7]
    for example should be the power-per-section of the i-th signal.

    Args:
        data (np.ndarray): Raw IQ Data of a single smartphone with shape (#signals, #samples_per_signal)
        transients (np.ndarray): Transient start- and end-points for this smartphone with shape (#signals, 2)

    Returns:
        np.ndarray: The extracted feature vectors with shape (#signals, #features).
    """
    _features = np.zeros((data.shape[0], len(FEATURE_NAMES)))
    ########## Your code starts here ##########
    for i in range(data.shape[0]):
        # Hole den relevanten Transientenbereich
        start, end = transients[i]
        if start == end:  # Überspringe, falls kein gültiger Transient erkannt wurde; sollte nicht vorkommen!
            continue

        transient_signal = data[i, start:end]

        amplitude = np.abs(transient_signal)
        phase = np.angle(transient_signal)

        # 1. Standard Deviation of Normalized Amplitude
        norm_amplitude = amplitude / np.max(amplitude)
        std_norm_amplitude = np.std(norm_amplitude)
        _features[i, 0] = std_norm_amplitude

        # 2. Standard Deviation of Normalized Phase
        norm_phase = phase / np.max(phase) 
        std_norm_phase = np.std(norm_phase)
        _features[i, 1] = std_norm_phase

        # 3. Standard Deviation of Normalized Frequency
        #cA, cD = pywt.dwt(amplitude, wavelet='db38')  # Approximation (cA) und Detailkoeffizienten (cD)
        norm_frequency = np.diff(phase) / (2 * np.pi)
        std_norm_frequency = np.std(norm_frequency)
        _features[i, 2] = std_norm_frequency

        # 4. Variance of Change in Amplitude
        diff_amplitude = np.diff(amplitude)
        variance_change_amplitude = np.std(diff_amplitude)
        _features[i, 3] = variance_change_amplitude

        # 5. Standard Deviation of Normalized In-Phase Data
        in_phase = transient_signal.real
        norm_in_phase = in_phase / np.max(in_phase)
        std_norm_in_phase = np.std(norm_in_phase)
        _features[i, 4] = std_norm_in_phase

        # 6. Standard Deviation of Normalized Quadrature Data
        quadrature = transient_signal.imag
        norm_quadrature = quadrature / np.max(quadrature)
        std_norm_quadrature = np.std(norm_quadrature)
        _features[i, 5] = std_norm_quadrature

        # 7. Standard Deviation of Mean-Centered Normalized Amplitude
        norm_mean_amplitude = amplitude / np.mean(amplitude)
        std_mean_centered_amp = np.std(norm_mean_amplitude)
        _features[i, 6] = std_mean_centered_amp

        # 8. Power per Section
        sec_window = 10 #-> sonst power von 350000
        power = amplitude ** 2
        sections = np.array_split(power, (power.shape[0] + sec_window-1)//sec_window)
        power_per_section = np.zeros(len(sections))
        k = 0
        for section in sections:
            power_per_section[k] = np.sum(section)
            k +=1
        std_power_per_section = np.std(power_per_section)
        _features[i, 7] = std_power_per_section

        # 9. Standard Deviation of Mean-Centered Normalized Phase
        mean_centered_phase = phase - np.mean(phase)
        norm_mean_phase = phase / np.max(mean_centered_phase)
        std_mean_centered_phase = np.std(norm_mean_phase)
        _features[i, 8] = std_mean_centered_phase

        # 10. Average Change in DWT Coefficients
        cA, cD = pywt.dwt(amplitude, wavelet='db38')
        avg_cA_diff = np.mean(np.abs(np.diff(cA)))
        _features[i, 9] = avg_cA_diff

        # 11. Length of Transient
        _features[i, 10] = end - start

        # 12. "Difference normalized mean and normalized maximum"
        _features[i, 11] = np.abs(np.mean(norm_amplitude) - np.max(norm_amplitude))
    ########## Your code ends here ##########
    return _features