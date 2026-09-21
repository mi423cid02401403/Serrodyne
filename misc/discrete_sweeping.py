import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from qick import *

# -----------------------------------------------------------------------------
# 1. HARDWARE & INITIALIZATION SETUP
# -----------------------------------------------------------------------------
soc = QickSoc()
soccfg = soc

GEN_CH = 1
RO_CH = 0

gencfg = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR = gencfg["f_fabric"] * samps_per_clk * 1e6  # envelope sample rate in Hz
ENV_MAXLEN = gencfg["maxlen"]                      # total envelope samples available on this channel

print(f"Generator {GEN_CH}: f_fabric={gencfg['f_fabric']:.3f} MHz, "
      f"samps_per_clk={samps_per_clk}, envelope sample rate={ENV_SR/1e9:.4f} GSPS")
print(f"Envelope memory available: {ENV_MAXLEN} samples "
      f"({ENV_MAXLEN/ENV_SR*1e6:.3f} us max total playback)")

# -----------------------------------------------------------------------------
# 2. SINGLE-TONE SERRODYNE STEP GENERATOR (continuous phase across steps)
# -----------------------------------------------------------------------------
def serrodyne_tone(freq_hz, duration_sec, sample_rate, amplitude, phase0=0.0, width=1.0):
    """One constant-frequency sawtooth segment. Returns (waveform, phase_at_end, n_samples).
    Carrying phase0 -> phase_at_end across successive calls keeps the ramp continuous
    when you jump from one step's envelope address to the next."""
    n_samples = max(1, int(round(float(duration_sec) * sample_rate)))
    dt = 1.0 / sample_rate
    t = np.arange(n_samples) * dt
    phase = 2 * np.pi * freq_hz * t + phase0
    y = float(amplitude) * scipy_signal.sawtooth(phase, width=width)
    phase_end = (phase0 + 2 * np.pi * freq_hz * n_samples * dt) % (2 * np.pi)
    return y, phase_end, n_samples

# -----------------------------------------------------------------------------
# 3. CHIRP DEFINITION (single tone, address-per-step)
# -----------------------------------------------------------------------------
F_STOP_HZ  = 76.25e6           # end of sweep (one of your original 4 tones)
CHIRP_SPAN_HZ = 350e6          # your target span
F_START_HZ = F_STOP_HZ - CHIRP_SPAN_HZ

NUM_STEPS = 30000                 # piecewise-constant frequency steps
AMPLITUDE = 0.9                # fraction of full scale, envelope-side

TOTAL_SWEEP_S_REQUESTED = 6e-3

# Hard ceiling from hardware: sum of all step lengths can't exceed ENV_MAXLEN.
# Leave a little headroom (95%) for samps_per_clk padding on each step.
max_feasible_total_s = 0.95 * ENV_MAXLEN / ENV_SR
TOTAL_SWEEP_S = min(TOTAL_SWEEP_S_REQUESTED, max_feasible_total_s)

if TOTAL_SWEEP_S < TOTAL_SWEEP_S_REQUESTED:
    print(f"WARNING: requested {TOTAL_SWEEP_S_REQUESTED*1e3:.3f} ms sweep needs "
          f"{TOTAL_SWEEP_S_REQUESTED*ENV_SR:.0f} envelope samples, but only "
          f"{ENV_MAXLEN} are available on gen {GEN_CH}.")
    print(f"Clamping this proof-of-concept to {TOTAL_SWEEP_S*1e6:.3f} us total "
          f"({NUM_STEPS} steps of {TOTAL_SWEEP_S/NUM_STEPS*1e9:.1f} ns each).")
    print("This validates the address-switching mechanism only — reaching 6 ms will "
          "need the periodic/addr-register approach (one period per step, hardware-repeated).")

step_duration_s = TOTAL_SWEEP_S / NUM_STEPS
step_duration_us = step_duration_s * 1e6

freqs_hz = np.linspace(F_START_HZ, F_STOP_HZ, NUM_STEPS)

maxv = soccfg.get_maxv(GEN_CH)
idata_list = []
qdata_list = []

phase = 0.0
for f in freqs_hz:
    y, phase, n_samples = serrodyne_tone(f, step_duration_s, ENV_SR, amplitude=AMPLITUDE, phase0=phase)

    pad = (-len(y)) % samps_per_clk
    if pad:
        y = np.concatenate([y, np.zeros(pad)])

    i_wave = np.round(y * maxv).astype(np.int16)
    q_wave = np.zeros_like(i_wave)

    idata_list.append(i_wave)
    qdata_list.append(q_wave)

samples_per_step = len(idata_list[0])
total_samples = samples_per_step * NUM_STEPS
print(f"Per-step length: {samples_per_step} samples ({samples_per_step/ENV_SR*1e9:.1f} ns)")
print(f"Total envelope samples requested: {total_samples} / {ENV_MAXLEN} available")
assert total_samples <= ENV_MAXLEN, (
    f"Envelope memory exceeded: need {total_samples} samples, "
    f"only {ENV_MAXLEN} available on gen {GEN_CH}. Reduce NUM_STEPS or TOTAL_SWEEP_S."
)

# -----------------------------------------------------------------------------
# 4. PROGRAM: switch envelope address every step, no periodic mode
# -----------------------------------------------------------------------------
class AddressSwitchedSerrodyneProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=1)

        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=cfg["readout_length"],
                                  freq=0, gen_ch=res_ch)

        for i, (idata_step, qdata_step) in enumerate(zip(cfg["idata_list"], cfg["qdata_list"])):
            self.add_envelope(ch=res_ch, name=f"serr_{i}", idata=idata_step, qdata=qdata_step)

        self.synci(200)

    def body(self):
        res_ch = self.cfg["res_ch"]

        # Each envelope already spans its full step duration, so t='auto'
        # (the default) chains them back-to-back with no manual sync_all
        # bookkeeping and no periodic mode — this is the actual address switch.
        for i in range(self.cfg["num_steps"]):
            self.set_pulse_registers(
                ch=res_ch,
                style="arb",
                freq=0,
                phase=0,
                gain=self.cfg["gain"],
                waveform=f"serr_{i}",
                outsel="input",
            )
            self.pulse(ch=res_ch, t='auto')

        self.sync_all()
        self.trigger(
            adcs=self.ro_chs,
            pins=[0],
            adc_trig_offset=self.cfg["adc_trig_offset"]
        )

# -----------------------------------------------------------------------------
# 5. EXECUTION & LOOPBACK CAPTURE
# -----------------------------------------------------------------------------
config = {
    "res_ch": GEN_CH,
    "ro_chs": [RO_CH],
    "reps": 1,
    "relax_delay": 1.0,
    "readout_length": min(1020, samples_per_step * NUM_STEPS),
    "adc_trig_offset": 100,
    "idata_list": idata_list,
    "qdata_list": qdata_list,
    "num_steps": NUM_STEPS,
    "step_duration_us": step_duration_us,
    "gain": 32767,
    "soft_avgs": 1,
}

prog = AddressSwitchedSerrodyneProgram(soccfg, config)
iq_list = prog.acquire_decimated(soc, progress=True)

plt.figure(figsize=(11, 3))
fs_iq = soccfg["readouts"][RO_CH]["f_output"] * 1e6
for ii, iq in enumerate(iq_list):
    t_us = np.arange(len(iq[0])) / fs_iq * 1e6
    plt.plot(t_us, iq[0], label=f"I, ADC {config['ro_chs'][ii]}")
    plt.plot(t_us, iq[1], label=f"Q, ADC {config['ro_chs'][ii]}")
plt.ylabel("a.u.")
plt.xlabel("Time (µs)")
plt.title("Address-switched serrodyne — loopback (watch step boundaries for glitches)")
plt.legend()
plt.savefig("/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/addr_switch_serrodyne.pdf", bbox_inches="tight")
plt.close()
