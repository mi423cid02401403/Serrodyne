import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from qick import *
from qick.averager_program import AveragerProgram

# -----------------------------------------------------------------------------
# 1. HARDWARE
# -----------------------------------------------------------------------------
soc    = QickSoc()
soccfg = soc

GEN_CH = 0
RO_CH  = 0

gencfg        = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR        = gencfg["f_fabric"] * samps_per_clk * 1e6

print(f"Generator {GEN_CH}: f_fabric={gencfg['f_fabric']:.3f} MHz, "
      f"samps_per_clk={samps_per_clk}, "
      f"envelope sample rate={ENV_SR/1e9:.4f} GSPS")

MAX_ENV_SECONDS = 6e-6
MAX_ENV_SAMPLES = int(MAX_ENV_SECONDS * ENV_SR)
READOUT_LEN     = 1020
fs_iq           = soccfg["readouts"][RO_CH]["f_output"] * 1e6
fft_res_hz      = fs_iq / READOUT_LEN

print(f"ADC IQ sample rate : {fs_iq/1e6:.3f} MHz")
print(f"ADC Nyquist limit  : {fs_iq/2/1e6:.3f} MHz")
print(f"FFT resolution     : {fft_res_hz/1e6:.4f} MHz")


# -----------------------------------------------------------------------------
# 2. HELPERS
# -----------------------------------------------------------------------------
def time_axis(num_samples, sample_rate):
    return np.arange(int(num_samples), dtype=float) / float(sample_rate)


def serrodyne(ratios, freqs_hz, total_seconds, *, amplitude, sample_rate,
              width=1.0, continuous_phase=False):
    if len(ratios) != len(freqs_hz):
        raise ValueError("ratios and frequencies must have same length")
    sample_rate = float(sample_rate)
    n_samples   = max(1, int(round(float(total_seconds) * sample_rate)))
    dt          = 1.0 / sample_rate
    ratio_sum   = sum(ratios)
    segment_lengths = [int(round(n_samples * (r / ratio_sum))) for r in ratios]
    delta = n_samples - sum(segment_lengths)
    idx   = 0
    while delta != 0 and idx < len(segment_lengths) * 4:
        si        = idx % len(segment_lengths)
        candidate = segment_lengths[si] + (1 if delta > 0 else -1)
        if candidate >= 0:
            segment_lengths[si] = candidate
            delta = n_samples - sum(segment_lengths)
        idx += 1

    y            = np.zeros(n_samples, dtype=float)
    start        = 0
    phase_offset = 0.0
    two_pi       = 2 * np.pi

    for length, frequency in zip(segment_lengths, freqs_hz):
        end = start + length
        if length > 0:
            t = time_axis(length, sample_rate)
            if frequency == 0:
                segment = np.zeros(length)
                if continuous_phase:
                    phase_offset = phase_offset % two_pi
            else:
                phase   = (two_pi * float(frequency) * t +
                           (phase_offset if continuous_phase else 0.0))
                segment = float(amplitude) * scipy_signal.sawtooth(
                    phase, width=float(width))
                if continuous_phase:
                    phase_offset = (two_pi * float(frequency) * (length * dt)
                                    + phase_offset) % two_pi
            y[start:end] = segment
        start = end
    return y, n_samples


def find_peak_near(spec, fft_freqs_hz, center_hz, search_bw_hz):
    """
    Find the FFT peak closest to center_hz within ±search_bw_hz.
    Returns the frequency of the peak in Hz.
    """
    mask        = np.abs(fft_freqs_hz - center_hz) < search_bw_hz
    spec_masked = np.where(mask, spec, 0)
    if spec_masked.max() == 0:
        # nothing found in window — fall back to global max
        print(f"  WARNING: no peak found within ±{search_bw_hz/1e6:.1f} MHz "
              f"of {center_hz/1e6:.3f} MHz — using global max")
        return fft_freqs_hz[np.argmax(spec)]
    return fft_freqs_hz[np.argmax(spec_masked)]


# -----------------------------------------------------------------------------
# 3. FREQUENCIES
# -----------------------------------------------------------------------------
TONE_END_HZ = np.array([
     76.25e6,
      0.00e6,
    (-122.92 + 76.25) * 1e6,
    (-147.82 + 76.25) * 1e6,
])

SWEEP_START_FRACTION    = 0.1
TONE_START_HZ           = TONE_END_HZ * SWEEP_START_FRACTION
ratios                  = [1, 1, 1, 1]
number_of_steps         = 20

freq_current_steps = np.array([
    np.linspace(TONE_START_HZ[i], TONE_END_HZ[i], number_of_steps)
    for i in range(len(TONE_END_HZ))
]).T   # (20, 4)

_max_hz = np.max(np.abs(freq_current_steps))
assert _max_hz <= 100e6, f"BUG: {_max_hz/1e6:.2f} MHz exceeds ±100 MHz"

demod_freq_mhz_per_step = freq_current_steps[:, 0] / 1e6

SHORT_PULSE_US = 2.0
STEP_ON_AIR_US = 250.0

n_samples_check = int(round(SHORT_PULSE_US * 1e-6 * ENV_SR))
assert n_samples_check <= MAX_ENV_SAMPLES, \
    f"SHORT_PULSE_US={SHORT_PULSE_US} us exceeds 6 us hardware limit"

highest_tone_hz = np.max(np.abs(TONE_END_HZ[TONE_END_HZ != 0]))
min_cycles      = (SHORT_PULSE_US * 1e-6) * highest_tone_hz
assert min_cycles >= 5, \
    f"Only {min_cycles:.1f} cycles in buffer. Increase SHORT_PULSE_US."

print(f"Buffer: {SHORT_PULSE_US} us = {n_samples_check} samples, "
      f"{min_cycles:.1f} cycles of highest tone")
print(f"Total sweep: {number_of_steps * STEP_ON_AIR_US / 1000:.3f} ms")

# ── search bandwidth for peak finding ────────────────────────────────────────
# Use ±half the step size in frequency so we never grab the wrong step's peak.
# Step size = (76.25 - 7.625) / 19 = 3.613 MHz → use ±3 MHz search window
freq_step_size_mhz = (demod_freq_mhz_per_step[-1]
                      - demod_freq_mhz_per_step[0]) / (number_of_steps - 1)
SEARCH_BW_HZ = max(3e6, freq_step_size_mhz * 0.4 * 1e6)
print(f"Frequency step size : {freq_step_size_mhz:.3f} MHz")
print(f"Peak search window  : ±{SEARCH_BW_HZ/1e6:.2f} MHz around expected")

maxv       = soccfg.get_maxv(GEN_CH)
idata_list = []
qdata_list = []

for i in range(number_of_steps):
    freq_step = freq_current_steps[i]
    y, _      = serrodyne(ratios, freq_step, SHORT_PULSE_US * 1e-6,
                          amplitude=0.12, sample_rate=ENV_SR,
                          continuous_phase=True, width=1.0)
    pad = (-len(y)) % samps_per_clk
    if pad:
        y = np.concatenate([y, np.zeros(pad)])
    assert len(y) <= MAX_ENV_SAMPLES
    idata_list.append(np.round(y * maxv).astype(np.int16))
    qdata_list.append(np.zeros(len(y), dtype=np.int16))

print(f"Generated {number_of_steps} envelopes ({len(idata_list[0])} samples).")


# -----------------------------------------------------------------------------
# 4. PROGRAM
# -----------------------------------------------------------------------------
class SingleStepSerrodyneProgram(AveragerProgram):
    def initialize(self):
        cfg    = self.cfg
        res_ch = cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=1)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                 freq=cfg["demod_freq_mhz"], gen_ch=res_ch)
        self.add_envelope(ch=res_ch, name="serrodyne",
                          idata=cfg["idata"], qdata=cfg["qdata"])
        self.set_pulse_registers(ch=res_ch, style="arb", freq=0, phase=0,
                                 gain=cfg["gain"], waveform="serrodyne",
                                 outsel="product", mode="periodic")
        self.synci(200)

    def body(self):
        self.pulse(ch=self.cfg["res_ch"])
        self.trigger(adcs=self.ro_chs, pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"])
        self.sync_all(self.us2cycles(2.0))


# -----------------------------------------------------------------------------
# 5. RUN
# -----------------------------------------------------------------------------
base_config = {
    "res_ch"          : GEN_CH,
    "ro_chs"          : [RO_CH],
    "reps"            : 1,
    "relax_delay"     : 1.0,
    "readout_length"  : READOUT_LEN,
    "adc_trig_offset" : 270,
    "gain"            : 32767,
    "soft_avgs"       : 10,
}

captures          = []
measured_peak_mhz = []

for i in range(number_of_steps):
    step_cfg = dict(base_config)
    step_cfg["idata"]           = idata_list[i]
    step_cfg["qdata"]           = qdata_list[i]
    step_cfg["demod_freq_mhz"]  = float(demod_freq_mhz_per_step[i])

    prog = SingleStepSerrodyneProgram(soccfg, step_cfg)
    iq   = prog.acquire_decimated(soc, progress=False)

    I = np.asarray(iq[0][0])
    Q = np.asarray(iq[0][1])
    captures.append((I, Q))

    z         = I + 1j * Q
    N         = len(z)
    spec      = np.abs(np.fft.fftshift(np.fft.fft(z)))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs_iq))

    # KEY FIX: search near expected frequency, not global argmax
    # The demod shifts the expected tone to DC (0 Hz offset),
    # so we search within SEARCH_BW_HZ of 0 Hz in the IQ FFT,
    # then convert back to absolute frequency
    peak_offset_hz = find_peak_near(
        spec,
        fft_freqs,                          # relative to demod (Hz)
        center_hz   = 0.0,                  # tone should be at DC
        search_bw_hz= SEARCH_BW_HZ,
    )
    peak_abs_mhz = demod_freq_mhz_per_step[i] + peak_offset_hz / 1e6
    measured_peak_mhz.append(peak_abs_mhz)

    error_mhz = peak_abs_mhz - demod_freq_mhz_per_step[i]
    print(f"Step {i:2d}: "
          f"expected={demod_freq_mhz_per_step[i]:7.3f} MHz  "
          f"measured={peak_abs_mhz:7.3f} MHz  "
          f"error={error_mhz:+.3f} MHz")

    # ── per-step plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))

    # envelope (first 200 ns)
    t_ns = np.arange(len(idata_list[i])) / ENV_SR * 1e9
    mask = t_ns <= 200
    axes[0].plot(t_ns[mask], idata_list[i][mask])
    axes[0].set_title(f"Step {i}: DAC envelope (first 200 ns)")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("DAC code")
    axes[0].grid(True, alpha=0.3)

    # IQ time trace
    t_us = np.arange(N) / fs_iq * 1e6
    axes[1].plot(t_us, I, label="I")
    axes[1].plot(t_us, Q, label="Q")
    axes[1].set_title(
        f"Step {i}: ADC  |  "
        f"expected={demod_freq_mhz_per_step[i]:.3f} MHz  "
        f"measured={peak_abs_mhz:.3f} MHz  "
        f"error={error_mhz:+.3f} MHz"
    )
    axes[1].set_xlabel("Time (us)")
    axes[1].set_ylabel("a.u.")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # FFT — show full spectrum AND search window
    freqs_abs = demod_freq_mhz_per_step[i] + fft_freqs / 1e6
    axes[2].plot(freqs_abs, spec / spec.max(), color="steelblue")
    axes[2].axvline(demod_freq_mhz_per_step[i],
                    color="red",   linestyle="--", linewidth=1.2,
                    label=f"Expected {demod_freq_mhz_per_step[i]:.2f} MHz")
    axes[2].axvline(peak_abs_mhz,
                    color="green", linestyle="--", linewidth=1.2,
                    label=f"Measured {peak_abs_mhz:.2f} MHz")
    axes[2].axvspan(demod_freq_mhz_per_step[i] - SEARCH_BW_HZ / 1e6,
                    demod_freq_mhz_per_step[i] + SEARCH_BW_HZ / 1e6,
                    alpha=0.15, color="red", label="Search window")
    axes[2].set_xlabel("Frequency (MHz)")
    axes[2].set_ylabel("Normalised magnitude")
    axes[2].set_title("FFT spectrum")
    axes[2].set_xlim(-100, 200)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/"
        f"serrodyne_step{i:02d}.pdf",
        bbox_inches="tight",
    )
    plt.close()

print("\nAll steps captured.")


# -----------------------------------------------------------------------------
# 6. SWEEP CONFIRMATION
# -----------------------------------------------------------------------------
expected_mhz     = demod_freq_mhz_per_step
measured_arr_mhz = np.array(measured_peak_mhz)
errors_mhz       = measured_arr_mhz - expected_mhz

print("\nExpected (MHz):", np.round(expected_mhz,     3))
print("Measured (MHz):", np.round(measured_arr_mhz,  3))
print("Error    (MHz):", np.round(errors_mhz,         3))
print(f"Max error      : {np.max(np.abs(errors_mhz)):.3f} MHz")
print(f"Mean error     : {np.mean(np.abs(errors_mhz)):.3f} MHz")
print(f"RMS error      : {np.sqrt(np.mean(errors_mhz**2)):.3f} MHz")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].plot(expected_mhz,     "o-", label="Expected tone 0")
axes[0].plot(measured_arr_mhz, "x-", label="Measured peak")
axes[0].axhline( 100, color="red", linestyle="--", linewidth=0.8)
axes[0].axhline(-100, color="red", linestyle="--", linewidth=0.8,
                label="±100 MHz limit")
axes[0].set_ylabel("Frequency (MHz)")
axes[0].set_title("Sweep: expected vs measured")
axes[0].set_ylim(-110, 110)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(number_of_steps), errors_mhz, color="orange")
axes[1].axhline(0,                color="black",  linewidth=0.8)
axes[1].axhline( SEARCH_BW_HZ/1e6, color="green",
                linestyle="--", linewidth=0.8,
                label=f"Search window ±{SEARCH_BW_HZ/1e6:.1f} MHz")
axes[1].axhline(-SEARCH_BW_HZ/1e6, color="green",
                linestyle="--", linewidth=0.8)
axes[1].set_ylabel("Error (MHz)")
axes[1].set_xlabel("Step index")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/"
    "serrodyne_sweep_confirmation.pdf",
    bbox_inches="tight",
)
plt.close()
print("Done.")