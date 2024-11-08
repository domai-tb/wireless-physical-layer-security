from pyhackrf2 import HackRF
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Print some more information to debug Useful for finding optimal parameters.
# E.g., a sample rate of 0.5e3 leads to an average time per measure ment of 0.45s
# but a sample rate of 1e3 was leading to an average time per measure ment of 0.22s.
# (Sample rate and numbers of samples to read was the same.)
DEBUG_PRINT = False

# Configuration
START_FREQ  = 75e6    # Start frequency in Hz
STOP_FREQ   = 1e9     # Stop frequency in Hz
STEP_SIZE   = 500e3   # Step size in Hz
SAMPLE_RATE = 20e6    # Sample Rate in Hz
SAMPLE_NUM  = 0.5e6   # Samples to read per measurement
SENSE_ITERATIONS = 20  # Number of iterations to measure power

# Derived Parameters
frequencies = np.arange(START_FREQ, STOP_FREQ, STEP_SIZE)
num_frequencies = len(frequencies)

# Initialize HackRF
hackrf = HackRF()
hackrf.amplifier_on = True
hackrf.lna_gain = 32
hackrf.vga_gain = 32

# Bandwith is automatically calculates 
# as 0.75 * SAMPLE_RATE by HackRF library
hackrf.sample_rate = SAMPLE_RATE

# Power Measurements
power_matrix = []

start_time = time()

try:
    for sense_i in range(SENSE_ITERATIONS):
        
        sense_start_time = time()
        power_row = []

        # Frequencies in STEP_SIZE Hz steps 
        for freq in frequencies:

            if DEBUG_PRINT:
                measure_start = time()
            
            # Measure samples at given frequence and bandwith 0.75 * SAMPLE_RATE
            hackrf.center_freq = freq
            samples = hackrf.read_samples(SAMPLE_NUM)

            # Calculate power based on measured samples
            power = 10 * np.log10(np.mean(np.abs(samples)**2))
            power_row.append(power)

            if DEBUG_PRINT:
                measure_time = time() - measure_start
                print(f"Frequency: {freq/1e6} MHz, Power: {power:.4f} dB, Time: {measure_time:.2f}s")
        
        power_matrix.append(power_row)  # Store power measurements for this sweep
        sense_iteration_time = time() - sense_start_time # Measure time for this sweep
        print(f"Completed iteration ({sense_i+1}/{SENSE_ITERATIONS}) in {sense_iteration_time:.2f}s")

finally:
    hackrf.close()

# Convert power data to numpy array for easy manipulation and plotting
power_matrix = np.array(power_matrix)

# Calculate the actual elapsed time
elapsed_time = time() - start_time

# Plotting
plt.figure(figsize=(12, 6))
plt.imshow(
    power_matrix, 
    aspect="auto",
    # We can assume that the time to measure the samples per frequence is nearly
    # constant / the same. So we can scale the time axis to the elapsed time here.
    # (Set `DEBUG_PRINT = True` to check assumption)
    extent=[START_FREQ / 1e6, STOP_FREQ / 1e6, 0, elapsed_time], 
    origin="lower", 
    cmap="viridis"
)
plt.colorbar(label="Power (dB)")
plt.ylabel("Time (s)")
plt.xlabel("Frequency (MHz)")
plt.title(f"Spectrogram from {START_FREQ / 1e6} MHz to {STOP_FREQ / 1e6} MHz")
plt.show()

# Signal detection:
#   A measurements is considered to be a signal if the power is
#   above the average easured power or the same. 
threshold = np.mean(power_matrix)
signals = power_matrix >= threshold 

plt.figure(figsize=(8, 6))
plt.imshow(
    signals,
    aspect="auto",
    extent=[START_FREQ / 1e6, STOP_FREQ / 1e6, 0, elapsed_time], 
    cmap=plt.get_cmap("Grays", 2), 
    interpolation="nearest", 
)
plt.title("Signals")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Time (s)")
plt.colorbar(ticks=[0,1], label="Signal Detection")
plt.clim(0,1)
plt.show()
