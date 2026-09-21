import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from math import gcd
from qick import *

# ── hardware ──────────────────────────────────────────────────────────────────
GEN_CH = 1
RO_CH  = 0
soc    = QickSoc()
soccfg = soc

gencfg        = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR        = gencfg["f_fabric"] * samps_per_clk * 1e6
MAXLEN        = gencfg["maxlen"]

print(f"ENV_SR={ENV_SR/1e9:.4f} GSPS, samps_per_clk={samps_per_clk}, maxlen={MAXLEN}")

# ── chirp parameters ──────────────────────────────────────────────────────────
ratios          = [0.337, 0.167, 0.288, 0.208]
freqs_hz        = np.array([-76.25e6, 0.0, +122.92e6 - 76.25e6, +147.82e6 - 76.25e6])
freq_hz_shifted = freqs_hz + 350e6

total_chirp_time = 6e-3   # 6 ms target
amp              = 1.0
width            = 1.0

# Sign convention
freq_start = -freq_hz_shifted   # start of chirp (per component)
freq_end   = -freqs_hz          # end of chirp   (per component)

# ── Step 1: compute the natural segment length for ONE step ──────────────────
# The composite segment = 4 sub-segments concatenated, each sub-seg = ratio * total
# The total length is driven by the LOWEST non-zero frequency sub-segment.
# We need to find the minimum seg_len (multiple of samps_per_clk) such that
# every sub-segment contains at least one full period of its frequency.

def min_seg_len_for_freqs(ratios, freqs_hz, sample_rate, samps_per_clk):
    """
    Find the smallest total segment length (multiple of samps_per_clk) such that
    each non-zero sub-segment (allocated by ratio) contains >= 1 full period.
    """
    ratio_sum = sum(ratios)
    min_total = samps_per_clk  # absolute floor

    for ratio, freq in zip(ratios, freqs_hz):
        if freq == 0.0:
            continue
        # samples needed for one period of this frequency
        one_period = sample_rate / abs(freq)
        # this sub-segment gets (ratio/ratio_sum) * total samples
        # we need: (ratio/ratio_sum) * total >= one_period
        # => total >= one_period * ratio_sum / ratio
        needed_total = one_period * ratio_sum / ratio
        # round up to multiple of samps_per_clk
        needed_total = int(np.ceil(needed_total / samps_per_clk)) * samps_per_clk
        min_total = max(min_total, needed_total)

    return min_total

# Compute for the worst-case step (lowest frequencies = start of chirp,
# since freq_start has the largest magnitudes after the sign flip)
worst_case_freqs = freq_start  # shape (4,)
natural_seg_len  = min_seg_len_for_freqs(ratios, worst_case_freqs, ENV_SR, samps_per_clk)

print(f"\nNatural segment length   : {natural_seg_len} samples "
      f"({natural_seg_len / ENV_SR * 1e9:.1f} ns)")

# ── Step 2: compute max chirped_number that fits in MAXLEN ───────────────────
max_chirped_number = MAXLEN // natural_seg_len
# Round down to a "nice" number
chirped_number = max_chirped_number
print(f"Max steps that fit       : {max_chirped_number}")
print(f"Using chirped_number     : {chirped_number}")

if chirped_number < 2:
    raise RuntimeError(
        f"Even 1 segment ({natural_seg_len} samples) is close to maxlen ({MAXLEN}). "
        f"Your frequencies are too low for this memory budget. "
        f"Consider using outsel='dds' and sweeping the freq register instead."
    )

dwell_us = (total_chirp_time / chirped_number) * 1e6
print(f"Dwell per step           : {dwell_us:.3f} µs")
print(f"Actual total chirp time  : {chirped_number * dwell_us / 1e3:.3f} ms")

# ── Step 3: build the chirp frequency table ───────────────────────────────────
freq_chirped = np.array([
    np.linspace(freq_start[i], freq_end[i], chirped_number)
    for i in range(4)
]).T   # shape (chirped_number, 4)

# ── Step 4: segment builder — quality-first ───────────────────────────────────
def build_segment_quality(ratios, freqs_hz, *, amplitude, sample_rate,
                           samps_per_clk, seg_len, phase_offset=0.0, width=1.0):
    """
    Build one composite serrodyne segment of exactly `seg_len` samples.
    Sub-segments are allocated by ratio. Each sub-segment gets at least
    samps_per_clk samples and contains >= 1 full period of its frequency.

    Returns: idata (int16), qdata (int16), seg_len (int), phase_out (float)
    """
    two_pi    = 2.0 * np.pi
    dt        = 1.0 / sample_rate
    ratio_sum = sum(ratios)

    # Allocate sub-segment lengths by ratio, rounded to samps_per_clk
    sub_lengths = []
    for ratio in ratios:
        n = int(round(seg_len * ratio / ratio_sum / samps_per_clk)) * samps_per_clk
        n = max(samps_per_clk, n)
        sub_lengths.append(n)

    # Fix rounding so total == seg_len exactly
    diff = seg_len - sum(sub_lengths)
    # Distribute diff in units of samps_per_clk to the largest sub-segments
    idx = np.argsort(sub_lengths)[::-1]
    i = 0
    while diff != 0:
        step = samps_per_clk if diff > 0 else -samps_per_clk
        candidate = sub_lengths[idx[i % len(idx)]] + step
        if candidate >= samps_per_clk:
            sub_lengths[idx[i % len(idx)]] = candidate
            diff -= step
        i += 1
        if i > len(sub_lengths) * 100:
            break  # safety

    y         = []
    phase_out = phase_offset

    for length, freq in zip(sub_lengths, freqs_hz):
        if freq == 0.0:
            y.append(np.zeros(length))
        else:
            t   = np.arange(length) * dt
            phi = two_pi * freq * t + phase_out
            seg = amplitude * scipy_signal.sawtooth(phi, width=width)
            y.append(seg)
            phase_out = (two_pi * freq * (length * dt) + phase_out) % two_pi

    y_out = np.concatenate(y)

    # Safety pad
    pad = (-len(y_out)) % samps_per_clk
    if pad:
        y_out = np.concatenate([y_out, np.zeros(pad)])

    maxv  = soccfg.get_maxv(GEN_CH)
    idata = np.round(y_out * maxv).astype(np.int16)
    qdata = np.zeros_like(idata)
    return idata, qdata, len(y_out), phase_out


# ── Step 5: build all segments ────────────────────────────────────────────────
print(f"\nBuilding {chirped_number} segments of {natural_seg_len} samples each …")
all_idata = []
all_qdata = []
phase_acc = 0.0

for i in range(chirped_number):
    idata, qdata, n, phase_acc = build_segment_quality(
        ratios, freq_chirped[i],
        amplitude=amp,
        sample_rate=ENV_SR,
        samps_per_clk=samps_per_clk,
        seg_len=natural_seg_len,
        phase_offset=phase_acc,
        width=width,
    )
    all_idata.append(idata)
    all_qdata.append(qdata)

seg_len_samples = natural_seg_len
seg_len_words   = seg_len_samples // samps_per_clk

# ── Step 6: pack ──────────────────────────────────────────────────────────────
packed_i = np.zeros(chirped_number * seg_len_samples, dtype=np.int16)
packed_q = np.zeros(chirped_number * seg_len_samples, dtype=np.int16)

for i, (id_, qd_) in enumerate(zip(all_idata, all_qdata)):
    s = i * seg_len_samples
    packed_i[s:s + len(id_)] = id_
    packed_q[s:s + len(qd_)] = qd_

total_samples = chirped_number * seg_len_samples
print(f"Total packed samples     : {total_samples} / {MAXLEN} "
      f"({100*total_samples/MAXLEN:.1f}% utilisation)")
print(f"seg_len_words            : {seg_len_words}")

# ── Step 7: plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9))

# (a) First 4 segments — time domain
t_full = np.arange(len(packed_i)) / ENV_SR * 1e9   # full time axis in ns
axes[0].plot(t_full, packed_i, lw=0.4, color='steelblue', alpha=0.7)

# for k in range(1, chirped_number):
#     axes[0].axvline(k * seg_len_samples / ENV_SR * 1e9,
#                     color='r', lw=0.8, ls='--', alpha=0.6)
axes[0].set_title("First 4 chirp segments — I data (red dashes = segment boundaries)")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("DAC code")
axes[0].set_xlim(5000,5500)
axes[0].grid(True, alpha=0.3)

# (b) Zoom into segment 0 only — should look like a clean sawtooth
t_seg0 = np.arange(seg_len_samples) / ENV_SR * 1e9
axes[1].plot(t_seg0, packed_i[:seg_len_samples], lw=0.8, color='C1')
axes[1].set_title("Segment 0 only — should be a clean composite sawtooth")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("DAC code")
axes[1].grid(True, alpha=0.3)

# (c) Spectrogram of full buffer
nfft = max(seg_len_samples, 8)
axes[2].specgram(packed_i.astype(float), Fs=ENV_SR / 1e6,
                 NFFT=nfft, noverlap=0, cmap="plasma")
axes[2].set_title("Spectrogram — frequency should sweep monotonically left→right")
axes[2].set_xlabel("Time (µs)")
axes[2].set_ylabel("Frequency (MHz)")
axes[2].set_ylim(0, 500)   # zoom to relevant band
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(
    "/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/chirp_packed_envelope.pdf",
    bbox_inches="tight",
)
plt.close()
print("Plot saved → chirp_packed_envelope.pdf")