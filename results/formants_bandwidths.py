# NOTE: Enter correct file name and sampling rate [fs]

import numpy as np

def get_formants_bw(audio_source, fs, num_formants):
    """
    Compute formant bandwidths (–3 dB points) of an audio signal.

    Parameters
    ----------
    audio_source : str or array_like
        If str, path to a text file of one sample per line.
        Otherwise, a 1-D array of time-domain samples.
    fs : float
        Sampling rate, in Hz.
    num_formants : int
        Number of formant bandwidths to extract.

    Returns
    -------
    bw_store : ndarray, shape (3, num_formants)
        Row 0: bandwidths (Hz)
        Row 1: formant center frequencies (Hz)
        Row 2: formant peak amplitudes (dB)
    """
    # --- load if filename given ---
    if isinstance(audio_source, str):
        audio_signal = np.loadtxt(audio_source)
    else:
        audio_signal = np.asarray(audio_source)

    N = audio_signal.size

    # ---- FFT & power in dB ----
    X = np.fft.fft(audio_signal)
    P_db = 10 * np.log10(np.abs(X)**2)

    # ---- Frequency axis matching MATLAB's n = N-1; f = 0:df:fs ----
    n = N - 1
    df = fs / n
    f = np.arange(0, fs + df, df)  # length = N

    bw_store = np.zeros((3, num_formants))
    count = 0
    i = 1  # start at second bin to allow P_db[i-1] comparison

    while count < num_formants and i < N - 1:
        # local peak test
        if P_db[i] > P_db[i - 1] and P_db[i] > P_db[i + 1]:
            f0 = f[i]
            A0 = P_db[i]
            A3 = A0 - 3  # –3 dB level

            # --- Left side: scan until drop below A3 ---
            j = i
            while j > 0 and P_db[j] > A3:
                j -= 1
            xL = f[j:j+2]
            yL = P_db[j:j+2]
            f_L = np.interp(A3, yL, xL)

            # --- Right side: scan until drop below A3 ---
            k = i
            while k < N - 1 and P_db[k] > A3:
                k += 1
            xR = f[k-1:k+1]
            yR = P_db[k-1:k+1]
            # note: interp requires increasing yR
            f_R = np.interp(A3, yR, xR)

            bw_store[0, count] = f_R - f_L
            bw_store[1, count] = f0
            bw_store[2, count] = A0

            count += 1
            i = k  # advance past this formant
        else:
            i += 1

    return bw_store


if __name__ == "__main__":
    import sys

    srate_mul = float(input("Enter the sampling rate multiplier : ").strip())

    # sample rate
    fs = 44100 * srate_mul
    fname = "recorded-pressure.txt"

    # Enter how many formants you want to print
    num_formants = 10

    bws = get_formants_bw(fname, fs, num_formants)

    for idx in range(bws.shape[1]):
        print(f"Formant {idx+1}:  frequency = {bws[1,idx]:.1f} Hz, "
              f"Amplitude = {bws[2,idx]:.1f} dB,  Bandwidth = {bws[0,idx]:.1f} Hz")
