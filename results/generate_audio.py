#!/usr/bin/env python3

import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

# ---- PARAMETERS ----
infile = 'recorded-pressure.txt' 
srate_mul = int(input("Enter the sampling rate multiplier : ").strip())
srate = 44100 * srate_mul

# ---- LOAD PRESSURE DATA ----
data = np.loadtxt('recorded-pressure.txt')
Pr_Audio = np.loadtxt(infile)

# ---- NORMALIZE & CLIP ----
maxP = np.max(np.abs(Pr_Audio))
audio = Pr_Audio / maxP
audio = np.clip(audio, -1.0, 1.0)

# ---- BUTTERWORTH LOW-PASS FILTER & DOWNSAMPLE ----
final = audio

if srate != 44100:
    # design a 2nd-order Butterworth low-pass at 1050 Hz
    fc = 5050.0               # cutoff frequency in Hz
    Wn = fc / (srate / 2.0)   # normalized cutoff (Nyquist = srate/2)
    b, a = butter(2, Wn, btype='low')

    # apply filter
    filtered = lfilter(b, a, audio)

    # downsample
    final = filtered[::srate_mul]

# ---- PRINT FINAL ARRAY ----
print("Final audio array values:")
print(audio)

# ---- WRITE OUT WAV ----
# WAV must be written at 44100 Hz regardless of srate
write('sound.wav', 44100, final.astype(np.float32))

print(f"Done: sound.wav ({len(final)} samples at 44 100 Hz).")
