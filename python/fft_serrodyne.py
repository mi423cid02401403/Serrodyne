import numpy as np
import matplotlib.pyplot as plt

f_saw = -300e6
fs = 9e9
T = 1000e-9
cutoff = 1e9 

N = int(T * fs)
t = np.arange(N) / fs

saw = 2 * ((f_saw * t + 0.5) % 1) - 1

fft_saw = np.fft.fft(saw)
freq = np.fft.fftfreq(N, d=1/fs)

fft_shifted = np.fft.fftshift(fft_saw)
freq_shifted = np.fft.fftshift(freq)

lowpass = np.abs(freq) <= cutoff

fft_filtered = fft_saw * lowpass

filtered_signal = np.fft.ifft(fft_filtered).real

fig, ax = plt.subplots(3, 1, figsize=(10, 10))

ax[0].plot(t * 1e9, saw)
ax[0].set_title("Original Sawtooth")
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Amplitude")
ax[0].set_xlim(0, 20)
ax[0].grid(True)

fft_magnitude = np.abs(fft_shifted) / N

ax[1].plot(freq_shifted / 1e9, fft_magnitude)
ax[1].axvline(cutoff / 1e9, linestyle="--", label="1 GHz cutoff")
ax[1].axvline(-cutoff / 1e9, linestyle="--")
ax[1].set_title("FFT of Sawtooth")
ax[1].set_xlabel("Frequency (GHz)")
ax[1].set_ylabel("Magnitude")
ax[1].set_xlim(-3, 3)
ax[1].grid(True)
ax[1].legend()

ax[2].plot(t * 1e9, filtered_signal, label="After 1 GHz LPF")
ax[2].plot(t * 1e9, saw, "--", alpha=0.5, label="Original")
ax[2].set_title("Signal After 1 GHz Low-Pass Filter")
ax[2].set_xlabel("Time (ns)")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlim(0, 20)
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()
