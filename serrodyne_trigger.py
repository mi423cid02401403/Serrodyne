
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

# print(soccfg)

GEN_CH = 0
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

# Generate a serrodyne waveform sized for this generator

gencfg = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR = gencfg["f_fabric"] * samps_per_clk * 1e6  # envelope sample rate, in Hz

print(f"Generator {GEN_CH}: f_fabric={gencfg['f_fabric']:.3f} MHz, "
      f"samps_per_clk={samps_per_clk}, envelope sample rate={ENV_SR/1e9:.4f} GSPS")

# Serrodyne segments: (ratio, frequency in Hz), same convention as 02_serrodyne.ipynb
# ratios = [0.337, 0.167, 0.288, 0.508] #0.208
ratios = [1,1,1,1] #0.208
freqs_hz = [76.25e6, 0, -122.92e6+76.25e6, -147.82e6+76.25e6]
# ratios = [1]
# freqs_hz = [50e6]
requested_total_seconds = 1.0e-6

x, y, n_samples = serrodyne(
    ratios,
    freqs_hz,
    requested_total_seconds,
    amplitude=0.12,
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

plt.savefig("/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/serrodyne_sending.pdf", bbox_inches="tight")
plt.close()

class SerrodyneProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=res_ch, nqz=1)

        # configure the readout(s) for loopback capture (optional, for verification)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                  freq=0, gen_ch=res_ch)

        # load the serrodyne waveform as an arbitrary envelope
        self.add_envelope(ch=res_ch, name="serrodyne",
                           idata=cfg["idata"], qdata=cfg["qdata"])

        # play the envelope directly (outsel="input"): no DDS multiplication,
        # since the serrodyne waveform already contains the desired ramp/frequency content
        self.set_pulse_registers(ch=res_ch, style="arb", freq=0, phase=0,
                                  gain=cfg["gain"], waveform="serrodyne",
                                  outsel="input", mode="periodic")

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        # Common time reference
        self.sync_all()

        # Define waveform delay after the trigger
        trigger_time = 0
        pulse_delay = self.cfg["pulse_delay"]

        # ADC trigger
        self.trigger(
            adcs=self.ro_chs,
            pins=[0],
            adc_trig_offset=self.cfg["adc_trig_offset"],
            t=trigger_time
        )

        # Start waveform after pulse_delay
        self.pulse(
            ch=self.cfg["res_ch"],
            t=trigger_time + pulse_delay
        )

        # Print the programmed delay
        print(f"Trigger time: {trigger_time} tProc cycles")
        print(f"Waveform start time: {trigger_time + pulse_delay} tProc cycles")
        print(f"Trigger -> waveform delay: {pulse_delay} tProc cycles")

        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))
        # this can do all the steps above all at once
        # self.measure(pulse_ch=self.cfg["res_ch"],
        #               adcs=self.ro_chs,
        #               pins=[0],
        #               adc_trig_offset=self.cfg["adc_trig_offset"],
        #               wait=True,
        #               syncdelay=self.us2cycles(self.cfg["relax_delay"]))

config = {
    "res_ch": GEN_CH,       # --Fixed
    "ro_chs": [RO_CH],      # --Fixed
    "reps": 1,              # --Fixed
    "relax_delay": 1.0,     # --us

    "readout_length": 800,  # [Clock ticks]
    # Try varying readout_length so the serrodyne pulse fits comfortably inside it

    "adc_trig_offset": 100, # [Clock ticks]

    "idata": idata,
    "qdata": qdata,
    "gain": 32767,          # [DAC units], scales the envelope further
    "num_periods": 1000,
    "soft_avgs": 1,
    "pulse_delay": 0,
}

prog = SerrodyneProgram(soccfg, config)
# print(prog)
iq_list = prog.acquire_decimated(soc, progress=True)
plt.figure(figsize=(11, 3))
fs_iq = soccfg["readouts"][RO_CH]["f_output"] * 1e6
for ii, iq in enumerate(iq_list):
    t_us = np.arange(len(iq[0])) / fs_iq * 1e6

    plt.plot(t_us, iq[0], label="I value, ADC %d" % config["ro_chs"][ii])
    plt.plot(t_us, iq[1], label="Q value, ADC %d" % config["ro_chs"][ii])

plt.ylabel("a.u.")
plt.xlabel("Time (µs)")
plt.title("Captured serrodyne pulse (loopback)")
plt.legend()

plt.show()

plt.savefig(
    "/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/serrodyne_recieved.pdf",
    bbox_inches="tight"
)
plt.close()

# Build complex IQ trace
iq = iq_list[0]

I = np.asarray(iq[0])
Q = np.asarray(iq[1])

z = I + 1j * Q
N = len(z)

# Effective sample rate of the captured IQ data
fs_iq = soccfg["readouts"][RO_CH]["f_output"] * 1e6  # 552.96 MHz

# FFT of complex IQ signal
spectrum = np.fft.fftshift(np.abs(np.fft.fft(z))) / N

# Frequency axis
freqs_mhz = (
    np.fft.fftshift(
        np.fft.fftfreq(N, d=1 / fs_iq)
    ) / 1e6
)

print(f"Number of samples: {N}")
print(f"IQ sample rate: {fs_iq / 1e6:.2f} MHz")
print(f"FFT resolution: {fs_iq / N / 1e6:.3f} MHz")
print(f"Frequency range: ±{fs_iq / 2 / 1e6:.2f} MHz")

# Plot
plt.figure(figsize=(11, 3))

plt.plot(freqs_mhz, spectrum)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title("Serrodyne spectrum — ADC loopback")

plt.grid(True, alpha=0.3)
plt.xlim(-150, 150)

plt.show()

#
# Ideal signal
signal = np.asarray(idata)
t_ns = np.asarray(t_ns)

N = len(signal)

# Sample spacing and sample rate
dt = (t_ns[1] - t_ns[0]) * 1e-9
fs = 1 / dt

# FFT
spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal))) / N

# Frequency axis
freqs_mhz = (
    np.fft.fftshift(
        np.fft.fftfreq(N, d=dt)
    ) / 1e6
)

print(f"Number of samples: {N}")
print(f"Sample spacing: {dt * 1e9:.3f} ns")
print(f"Sample rate: {fs / 1e6:.2f} MHz")
print(f"FFT resolution: {fs / N / 1e6:.3f} MHz")
print(f"Frequency range: ±{fs / 2 / 1e6:.2f} MHz")

# Plot
plt.figure(figsize=(11, 3))

plt.plot(freqs_mhz, spectrum)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title("Ideal serrodyne spectrum")

plt.grid(True, alpha=0.3)
plt.xlim(-150, 150)

plt.show()

def plot_eom_spectrum(phase_codes, sample_rate, title="Expected EOM spectrum"):
    data = np.asarray(phase_codes, dtype=float)

    span = float(np.max(data) - np.min(data))
    if span == 0.0:
        raise ValueError("Cannot plot spectrum for a constant waveform")

    # Convert waveform to instantaneous phase [rad] (this nonlinear remap is what lets us
    # see the correct sign, unlike a plain magnitude FFT of the real captured samples)
    phase = 2 * np.pi * (data - np.min(data)) / span

    # Complex field after phase modulation
    field = np.exp(1j * phase)

    N = len(field)

    # FFT
    spectrum = np.fft.fftshift(
        np.abs(np.fft.fft(field))
    ) / N

    # Frequency axis
    freqs_mhz = (
        np.fft.fftshift(
            np.fft.fftfreq(N, d=1 / sample_rate)
        ) / 1e6
    )

    print(f"Number of samples: {N}")
    print(f"Sample rate: {sample_rate / 1e9:.3f} GS/s")
    print(f"FFT resolution: {sample_rate / N / 1e6:.3f} MHz")
    print(f"Frequency range: \u00b1{sample_rate / 2 / 1e6:.2f} MHz")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 3))

    ax.plot(freqs_mhz, spectrum)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-350, 350)  # widened so harmonics of the -100 MHz segment are visible too

    plt.show()

    return ax

I_captured = np.asarray(iq_list[0][0], dtype=float)
threshold = 0.2 * np.max(np.abs(I_captured))
active_idx = np.where(np.abs(I_captured) > threshold)[0]
if len(active_idx) == 0:
    raise RuntimeError("No samples above threshold -- check loopback cabling / gain / threshold.")
i0, i1 = active_idx[0], active_idx[-1] + 1
I_active = I_captured[i0:i1]

print(f"Active window: samples [{i0}:{i1}] out of {len(I_captured)} captured "
      f"({len(I_active)} active, {len(I_captured) - len(I_active)} dead-time samples trimmed)")

fs_readout = soccfg["readouts"][RO_CH]["f_output"] * 1e6

plot_eom_spectrum(
    I_active,
    fs_readout,
    title="Expected serrodyne EOM spectrum (active samples only)"
)
plt.savefig("/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/serrodyne_fft_shifts.pdf", bbox_inches="tight")
plt.close()