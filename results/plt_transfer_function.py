# NOTE: Enter correct file name and sampling rate [fs]

# Plot transfer function
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from recorded-pressure.txt
def read_recorded_pressure(file_name):

    """Read from recorded-pressure.txt"""
    with open(file_name, 'r') as f:
        # Strip whitespace, skip empty lines, convert to float
        data = [float(line.strip()) for line in f if line.strip()]
    return data

if __name__ == "__main__":
    filename = "recorded-pressure.txt"
    pressures = read_recorded_pressure(filename)

    N = len(pressures)
    audio_fft = np.fft.fft(pressures)

    transfer_function = audio_fft
    transfer_function_db = 10 * np.log10(np.abs(transfer_function)**2)

    srate_mul = float(input("Enter the sampling rate multiplier : ").strip())
    fs = 44100 * srate_mul #sample rate
    df = fs / N
    f = np.linspace(0, fs, N, endpoint=False)

    plt.figure(figsize=(8, 4))
    plt.plot(f, transfer_function_db, linewidth=1.5)
    plt.title('Amplitude Spectrum Analysis', fontweight='bold')
    plt.xlabel('Frequency [Hz]', fontweight='bold')
    plt.ylabel('Amplitude [dB]', fontweight='bold')
    plt.xlim(40, 14000)
    plt.ylim(auto=True)  # let matplotlib choose y-limits; remove if you want manual control
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()