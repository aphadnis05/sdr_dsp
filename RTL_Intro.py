import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load WAV file
samplerate, audio = wavfile.read("baseband_93385000Hz_13-24-49_04-09-2025.wav")

print("Sample rate:", samplerate)
print("Shape:", audio.shape)

# If stereo, take one channel
if audio.ndim > 1:
    audio = audio[:,0]

# Normalize to float
audio = audio / np.max(np.abs(audio))

# Plot waveform
plt.figure(figsize=(12,4))
plt.plot(audio[:5000])
plt.title("Audio waveform (first 5000 samples)")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.show()

# Plot spectrum
fft = np.fft.fft(audio[:4096])
freqs = np.fft.fftfreq(len(fft), 1/samplerate)

plt.figure(figsize=(10,6))
plt.semilogy(freqs[:len(freqs)//2], np.abs(fft[:len(freqs)//2]))
plt.title("Spectrum of audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
