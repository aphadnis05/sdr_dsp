import numpy as np
from scipy.io import wavfile
from scipy import signal
from fractions import Fraction

# -------- CONFIG --------
INPUT_FILE = "baseband_93385000Hz_14-34-23_19-09-2025.wav"   # your baseband file
OUTPUT_FILE = "demod_audio.wav"
TARGET_AUDIO_FS = 48000       # audio rate for output .wav
AUDIO_CUTOFF_HZ = 15000.0     # FM broadcast audio cutoff
DEEMPH_TAU = 75e-6            # 75 us de-emphasis (US standard)
# -------------------------

# ---- Load IQ file ----
fs, data = wavfile.read(INPUT_FILE)
print(f"Loaded {INPUT_FILE} -> {data.shape} samples/channels @ {fs} Hz")

if data.ndim < 2:
    raise ValueError("Need stereo IQ file (I=left, Q=right)!")

i = data[:,0].astype(np.float32)
q = data[:,1].astype(np.float32)

# Normalize if integers
if np.issubdtype(data.dtype, np.integer):
    i /= np.iinfo(data.dtype).max
    q /= np.iinfo(data.dtype).max

iq = i + 1j*q

# ---- FM demodulation ----
prod = iq[1:] * np.conj(iq[:-1])
dphi = np.angle(prod)
demod = dphi * (fs / (2*np.pi))  # instantaneous freq
demod -= np.mean(demod)          # DC removal

# ---- Resample to audio ----
fr = Fraction(TARGET_AUDIO_FS, fs).limit_denominator()
up, down = fr.numerator, fr.denominator
audio_resampled = signal.resample_poly(demod, up, down)
audio_fs = int(round(fs * up / down))
print(f"Resampled -> {audio_fs} Hz")

# ---- Low-pass filter ----
nyq = 0.5 * audio_fs
cutoff = AUDIO_CUTOFF_HZ / nyq
b = signal.firwin(401, cutoff)
audio_filt = signal.filtfilt(b, [1.0], audio_resampled)

# ---- De-emphasis ----
a = np.exp(-1.0 / (audio_fs * DEEMPH_TAU))
b0 = 1.0 - a
deemph = np.zeros_like(audio_filt)
deemph[0] = audio_filt[0]
for n in range(1, len(audio_filt)):
    deemph[n] = a*deemph[n-1] + b0*audio_filt[n]

# ---- Normalize & save ----
deemph /= np.max(np.abs(deemph)) + 1e-12
out = (deemph * 32767).astype(np.int16)

wavfile.write(OUTPUT_FILE, audio_fs, out)
print(f"✅ Wrote {OUTPUT_FILE} ({len(out)} samples @ {audio_fs} Hz)")
