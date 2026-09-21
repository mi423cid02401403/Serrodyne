
import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qick import *
# %pylab inline
# Load bitstream with custom overlay
soc = QickSoc()
# Since we're running locally on the QICK, we don't need a separate QickConfig object.
# If running remotely, you could generate a QickConfig from the QickSoc:
#     soccfg = QickConfig(soc.get_cfg())
# or save the config to file, and load it later:
#     with open("qick_config.json", "w") as f:
#         f.write(soc.dump_cfg())
#     soccfg = QickConfig("qick_config.json")
soccfg = soc

print(soccfg)

GEN_CH = 1
RO_CH = 1

from scipy import signal as scipy_signal


def time_axis(num_samples, sample_rate):
    """Return a time axis in seconds."""
    return np.arange(int(num_samples), dtype=float) / float(sample_rate)


def serrodyne(
    ratios,
    freqs_hz,
    total_seconds,
    *,
    amplitude,
    sample_rate,
    max_points=None,
    width=1.0,
    continuous_phase=False,
):
    """Generate a piecewise serrodyne sawtooth waveform.

    Ported from firmware.signals.serrodyne (RFSoC4x2 AWG overlay).
    """
    if len(ratios) != len(freqs_hz):
        raise ValueError("ratios and frequencies must have same length")
    if total_seconds <= 0:
        raise ValueError("total_seconds must be > 0")
    if not (0.0 <= width <= 1.0):
        raise ValueError("width must be in [0, 1]")

    sample_rate = float(sample_rate)
    n_samples = max(1, int(round(float(total_seconds) * sample_rate)))
    if max_points is not None:
        n_samples = min(n_samples, int(max_points))
    dt = 1.0 / sample_rate

    ratio_sum = sum(ratios)
    segment_lengths = [int(round(n_samples * (ratio / ratio_sum))) for ratio in ratios]
    delta = n_samples - sum(segment_lengths)
    index = 0
    while delta != 0 and index < len(segment_lengths) * 4:
        segment_index = index % len(segment_lengths)
        candidate = segment_lengths[segment_index] + (1 if delta > 0 else -1)
        if candidate >= 0:
            segment_lengths[segment_index] = candidate
            delta = n_samples - sum(segment_lengths)
        index += 1

    x = time_axis(n_samples, sample_rate)
    y = np.zeros(n_samples, dtype=float)
    start = 0
    phase_offset = 0.0
    two_pi = 2 * np.pi

    for length, frequency in zip(segment_lengths, freqs_hz):
        end = start + length
        if length > 0:
            t = time_axis(length, sample_rate)
            if frequency == 0:
                segment = np.zeros(length)
                if continuous_phase:
                    phase_offset = phase_offset % two_pi
            else:
                phase = (two_pi * float(frequency) * t) + (phase_offset if continuous_phase else 0.0)
                segment = float(amplitude) * (scipy_signal.sawtooth(phase, width=float(width))) 
                if continuous_phase:
                    phase_offset = (two_pi * float(frequency) * (length * dt) + phase_offset) % two_pi
            y[start:end] = segment
        start = end

    return x, y, n_samples

# Generate a serrodyne waveform sized for this generator

gencfg = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR = gencfg["f_fabric"] * samps_per_clk * 1e6  # envelope sample rate, in Hz

print(f"Generator {GEN_CH}: f_fabric={gencfg['f_fabric']:.3f} MHz, "
      f"samps_per_clk={samps_per_clk}, envelope sample rate={ENV_SR/1e9:.4f} GSPS")

# Serrodyne segments: (ratio, frequency in Hz), same convention as 02_serrodyne.ipynb
ratios = [0.337, 0.167, 0.288, 0.208]
actual_freq_shift = np.array([-76.25e6, 0, +122.92e6-76.25e6, +147.82e6-76.25e6])
amp = 0.1
freqs_hz = actual_freq_shift*(-1) # due to signal being flipped when measured
requested_total_seconds = 1.0e-6

x, y, n_samples = serrodyne(
    ratios,
    freqs_hz,
    requested_total_seconds,
    amplitude=amp,
    sample_rate=ENV_SR,
    continuous_phase=True,
)

# Zero-pad to a valid envelope length (multiple of samps_per_clk)
pad = (-len(y)) % samps_per_clk
if pad:
    y = np.concatenate([y, np.zeros(pad)])

actual_total_seconds = len(y) / ENV_SR

print(f"Requested serrodyne period: {requested_total_seconds * 1e6:.6f} us")
print(f"Active samples: {n_samples}, padded to {len(y)} (multiple of {samps_per_clk})")
print(f"Actual serrodyne period: {actual_total_seconds * 1e6:.6f} us")

# Convert to int16 envelope data, scaled to this generator's max envelope amplitude
maxv = soccfg.get_maxv(GEN_CH)
idata = np.round(y * maxv).astype(np.int16)
qdata = np.zeros_like(idata)

fig, ax = plt.subplots(figsize=(11, 3))
t_ns = np.arange(len(idata)) / ENV_SR * 1e9
ax.plot(t_ns, idata)
ax.set_title("Serrodyne envelope (I data) to be loaded into the generator")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("DAC code")
ax.grid(True, alpha=0.3)

plt.savefig("/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/serrodyne_envelope.png",
            bbox_inches="tight")
plt.close()
