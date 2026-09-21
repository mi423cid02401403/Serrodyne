import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qick import *

soc = QickSoc()
soccfg = soc

GEN_CH = 1
RO_CH = 0

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


gencfg = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR = gencfg["f_fabric"] * samps_per_clk * 1e6  # envelope sample rate, in Hz

print(f"Generator {GEN_CH}: f_fabric={gencfg['f_fabric']:.3f} MHz, "
      f"samps_per_clk={samps_per_clk}, envelope sample rate={ENV_SR/1e9:.4f} GSPS")

# ---------------------------------------------------------
# Waveform 1
# ---------------------------------------------------------
# ratios = [0.337, 0.167, 0.288, 5]  # 0.208
# freqs_hz = np.array([-76.25e6, 0, +122.92e6 - 76.25e6, +147.82e6 - 76.25e6])
freqs_hz = np.array([25e6])
ratios = [1]
amp = 1
freqs_hz *= -1
requested_total_seconds = 1e-6

x, y, n_samples = serrodyne(
    ratios,
    freqs_hz,
    requested_total_seconds,
    amplitude=amp,
    sample_rate=ENV_SR,
    continuous_phase=True,
)

pad = (-len(y)) % samps_per_clk
if pad:
    y = np.concatenate([y, np.zeros(pad)])

actual_total_seconds = len(y) / ENV_SR
print(f"Requested serrodyne period: {requested_total_seconds * 1e6:.6f} us")
print(f"Active samples: {n_samples}, padded to {len(y)} (multiple of {samps_per_clk})")
print(f"Actual serrodyne period: {actual_total_seconds * 1e6:.6f} us")

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
plt.savefig("/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/serrodyne_sending.pdf", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Waveform 2: same segment ratios/lengths, every frequency shifted
# +1 MHz relative to waveform 1 (freqs_hz here is already negated above,
# so this genuinely moves each tone up by 1 MHz in real frequency)
# ---------------------------------------------------------
freqs_hz_2 = freqs_hz - 50e6  # freqs_hz was negated above; subtracting shifts the *real* freq up by 1 MHz

x2, y2, n_samples2 = serrodyne(
    ratios,
    freqs_hz_2,
    requested_total_seconds,
    amplitude=amp,
    sample_rate=ENV_SR,
    continuous_phase=True,
)

pad2 = (-len(y2)) % samps_per_clk
if pad2:
    y2 = np.concatenate([y2, np.zeros(pad2)])

idata2 = np.round(y2 * maxv).astype(np.int16)
qdata2 = np.zeros_like(idata2)

# Keep bookkeeping trivial: put waveform 2 immediately after waveform 1
# in envelope memory (this only works cleanly if the padded lengths match;
# if they don't, just use addr2 = len(idata) regardless -- that's still a
# valid, non-overlapping address, it's just not adjacent to a "boundary").
print(f"Waveform 1 length: {len(idata)} samples")
print(f"Waveform 2 length: {len(idata2)} samples")

addr2 = len(idata)
required_len = addr2 + len(idata2)
if required_len > gencfg["maxlen"]:
    raise RuntimeError(
        f"waveform 2 needs envelope memory up to sample {required_len}, "
        f"but generator {GEN_CH} only has {gencfg['maxlen']} samples available"
    )

# ---------------------------------------------------------
# Program: trigger waveform 1 once, mode="periodic" so it keeps looping
# on the DAC with no further tProc involvement. run() is non-blocking --
# it loads the program+envelope, triggers, and returns immediately.
# ---------------------------------------------------------
class ChirpTestProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=1)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                  freq=0, gen_ch=res_ch)
        self.add_envelope(ch=res_ch, name="serrodyne",
                           idata=cfg["idata"], qdata=cfg["qdata"])
        self.set_pulse_registers(ch=res_ch, style="arb", freq=0, phase=0,
                                  gain=cfg["gain"], waveform="serrodyne",
                                  outsel="input", mode="periodic")
        self.synci(200)

    def body(self):
        self.pulse(ch=self.cfg["res_ch"], t=0)
        self.sync_all()


config = {
    "res_ch": GEN_CH,
    "ro_chs": [RO_CH],
    "reps": 1,
    "relax_delay": 0,
    "readout_length": 1020,
    "adc_trig_offset": 10,
    "idata": idata,
    "qdata": qdata,
    "gain": 1000,
    "soft_avgs": 1,
}

prog1 = ChirpTestProgram(soccfg, config)

# non-blocking: loads program + envelope 1, triggers it, returns immediately
prog1.run(soc)
print("Waveform 1 triggered and playing periodically (uninterrupted).")

# Pack I/Q into the (N, 2) int16 shape load_envelope() expects
data2 = np.column_stack([idata2, qdata2]).astype(np.int16)

time.sleep(2)
t0 = time.perf_counter()
soc.load_envelope(GEN_CH, data=data2, addr=addr2)
t1 = time.perf_counter()

upload_time_ms = (t1 - t0) * 1e3
n_bytes = data2.nbytes
print(f"Uploaded waveform 2 ({len(idata2)} samples, {n_bytes} bytes) "
      f"in {upload_time_ms:.3f} ms "
      f"({n_bytes / (t1 - t0) / 1e6:.2f} MB/s)")
print("Waveform 1 should have kept playing on the DAC throughout this upload, "
      "since it was never stopped and waveform 2 was written to a "
      "non-overlapping address.")