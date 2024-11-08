import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import time

# Function to perform repeated scans and create a time-frequency waterfall plot
def spectrum_sensing(start_freq, end_freq, bandwidth, sample_rate, step_size, num_scans):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.gain = 'auto'
    
    # Frequency steps
    freq_steps = np.arange(start_freq, end_freq, step_size)
    num_freqs = len(freq_steps)
    
    # Initialize array to store power measurements over time
    power_matrix = np.zeros((num_scans, num_freqs))
    
    # Start scanning over time
    for scan_idx in range(num_scans):
        for freq_idx, freq in enumerate(freq_steps):
            sdr.center_freq = freq
            samples = sdr.read_samples(1024)
            
            # Calculate power in dB for each frequency step
            power = np.mean(np.abs(np.fft.fft(samples))**2)
            power_db = 10 * np.log10(power)
            
            # Store the power value in the matrix
            power_matrix[scan_idx, freq_idx] = power_db
            
            # Short delay between frequency steps for faster scanning
            time.sleep(0.01)  # Reduced to 10 ms
        
        print(f"Completed scan {scan_idx + 1} of {num_scans}")
        # Short delay between scans
        time.sleep(0.05)
    
    sdr.close()
    return freq_steps, power_matrix

# Set parameters
start_freq = 75e6    # 75 MHz
end_freq = 1e9       # 1 GHz
bandwidth = 1e6      # 1 MHz step size for faster scanning
sample_rate = 2.048e6  # 2.048 MHz
step_size = bandwidth
num_scans = 20       # Fewer scans for quicker total time

# Run the time-frequency scan
freq_steps, power_matrix = spectrum_sensing(start_freq, end_freq, bandwidth, sample_rate, step_size, num_scans)

# Plot the waterfall (time-frequency) plot
plt.figure(figsize=(12, 6))
plt.imshow(power_matrix, aspect='auto', extent=[start_freq / 1e6, end_freq / 1e6, 0, num_scans],
           cmap='viridis', origin='lower')
plt.colorbar(label='Power (dB)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Time (scan index)')
plt.title('Time-Frequency Waterfall Plot')
plt.show()
