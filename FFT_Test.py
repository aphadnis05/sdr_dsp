import numpy as np
import matplotlib.pyplot as plt

fs = 1000  # Sampling rate, Hz
t = np.arange(0, 1, 1/fs)  # 1 second of samples
f = 5  # Frequency of sine wave, Hz

signal = np.sin(2 * np.pi * f * t)
noisy_signal = signal + 0.2*np.random.randn(len(signal))#np.sin(2 * np.pi * 2 * t)


plt.plot(t, noisy_signal)
plt.title("Time-domain sine wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

#Frequency-domain representation using FFT
fft_signal = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(fft_signal), 1/fs)

fft_noisy_signal = np.fft.fft(noisy_signal)
freqs = np.fft.fftfreq(len(fft_noisy_signal), 1/fs)

plt.plot(freqs[:len(freqs)//2], np.abs(fft_noisy_signal)[:len(freqs)//2])
plt.title("Frequency-domain (FFT) of sine wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0,10 ) 
plt.ylim(0,1000 ) # Show only positive frequencies
plt.show()



#Adding noise and filtering
from scipy.signal import butter, lfilter

# Adding random noise
noisy_signal = signal + 0.5*np.random.randn(len(signal))

# Simple low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff / (0.5*fs), btype='low')
    y = lfilter(b, a, data)
    return y

filtered_signal = butter_lowpass_filter(noisy_signal, cutoff=10, fs=fs)

plt.plot(t, noisy_signal, alpha=0.5, label='Noisy')
plt.plot(t, filtered_signal, label='Filtered')
plt.legend()
plt.xlim(0,1)
plt.show()
