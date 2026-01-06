import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate

import matplotlib.pyplot as plt


INPUT_FILE = "baseband_93385000Hz_14-34-23_19-09-2025.wav"   # Your recorded IQ file
OUTPUT_FILE = "fm_audio.wav"  # Demodulated audio file
AUDIO_RATE = 48000            # Final audio sample rate (Hz)

# --------------------------
# LOAD BASEBAND IQ DATA
# --------------------------
fs, data = wavfile.read(INPUT_FILE)
print( "Initial FS:")
print(fs)
# Ensure data is complex (I + jQ)
if data.ndim == 2:
    iq = data[:,0] + 1j*data[:,1]   # stereo channels = I/Q
else:
    raise ValueError("Expected stereo IQ .wav file (I=left, Q=right)")

iq = iq.astype(np.complex64)

print(f"Loaded {INPUT_FILE}: {len(iq)} samples @ {fs/1e6:.2f} MHz")

# --------------------------
# FM DEMODULATION
# --------------------------
phase = np.angle(iq)
unwrapped = np.unwrap(phase)
dphase = np.diff(unwrapped)

fm_demod = dphase * fs / (2*np.pi)   # instantaneous freq in Hz

# --------------------------
# DECIMATE TO AUDIO RATE
# --------------------------
decimation = int(fs // AUDIO_RATE)
if decimation > 1:
    audio = decimate(fm_demod, decimation)
else:
    audio = fm_demod

# Normalize audio to [-1, 1]
audio /= np.max(np.abs(audio))

# --------------------------
# SAVE AUDIO FILE
# --------------------------
wavfile.write(OUTPUT_FILE, AUDIO_RATE, audio.astype(np.float32))
print(f"Done! Saved demodulated audio to {OUTPUT_FILE}")


# Replace 'your_audio_file.wav' with the name of your file
sampling_rate, audio_data = wavfile.read('fm_audio.wav')
print( "sampling rate of demod ")
print(sampling_rate)
from scipy.signal import butter, lfilter

cutoff_freq_low = 300.0  # Hz
nyquist_freq = 0.5 * sampling_rate

# Design the filter
b, a = butter(5, cutoff_freq_low / nyquist_freq, btype='high')
# Apply the high-pass filter to the audio data
filtering_data = lfilter(b, a, audio_data)
# Perform the FFT on the audio data

cutoff_freq_high = 3000.0  # Hz
nyquist_freq = 0.5 * sampling_rate

# Design the filter
c, d = butter(5, cutoff_freq_high / nyquist_freq, btype='low')
# Apply the high-pass filter to the audio data
filtered_data = lfilter(c, d, filtering_data)
fft_output = np.fft.fft(filtered_data)

# Calculate the magnitude of the FFT output
# The FFT output is a complex array, so we take the absolute value
fft_magnitude = np.abs(fft_output)

# Get the number of samples
N = len(filtered_data)

# Get the frequencies in Hertz (Hz)
# The second argument is the inverse of the sampling rate (the time step between samples)
frequencies = np.fft.fftfreq(N, 1.0/sampling_rate)

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:N//2], fft_magnitude[:N//2]) # We only plot the first half
plt.title('Frequency Spectrum of Audio File')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0,1000 ) 
plt.ylim(0,20000 )
plt.show()




output_file = 'filtered_audio.wav'

# Write the data to the new file
wavfile.write(output_file, sampling_rate, filtered_data.astype(np.float32))