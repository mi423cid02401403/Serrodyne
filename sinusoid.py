import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")

import time
from qick import *

# ---------------------------------------------------------
# Connect to RFSoC
# ---------------------------------------------------------
soc = QickSoc()
soccfg = soc

# GEN_CH=0 → DAC_B (axis_signal_gen_v6, fs=9830.4 Msps)
GEN_CH = 0

# ---------------------------------------------------------
# Hardware limit — DAC gain is a signed value, max positive
# code is 32766 (full-scale output).
# ---------------------------------------------------------
MAX_GAIN = 32766

# ---------------------------------------------------------
# User parameters — vary these to change the transmitted tone
# ---------------------------------------------------------
frequency_mhz = 50.0        # Signal frequency (MHz)
pulse_gain    = MAX_GAIN    # DAC gain, 0-32766 (32766 = max DAC output)

# How long to leave the CW tone running before the script stops it.
# The waveform itself is continuous/periodic in hardware — this is
# just how long the Python side waits before disabling the output.
run_time_s = 1000.0

print(f"Frequency:   {frequency_mhz} MHz")
print(f"Gain:        {pulse_gain} / {MAX_GAIN} max")


class SendContinuousSinusoidProgram(AveragerProgram):
    """Transmit-only program: plays a const (CW) pulse out of the DAC
    in 'periodic' mode. Once triggered, the DAC keeps outputting the
    waveform continuously in hardware — it does NOT stop when the
    tProc instruction stream finishes. No ADC readout is configured;
    this only sends, it doesn't listen."""

    def initialize(self):
        cfg = self.cfg
        gen_ch = cfg["gen_ch"]

        self.declare_gen(ch=gen_ch, nqz=1)

        self.set_pulse_registers(
            ch=gen_ch,
            style="const",
            freq=self.freq2reg(cfg["frequency_mhz"], gen_ch=gen_ch),
            phase=0,
            gain=cfg["pulse_gain"],
            length=self.us2cycles(3.0, gen_ch=gen_ch),  # length of one period-repeat unit; irrelevant in periodic mode beyond keeping it short
            mode="periodic",   # <-- key change: loop the waveform continuously in hardware
        )

        self.synci(200)

    def body(self):
        cfg = self.cfg
        # Trigger once — in periodic mode the DAC continues outputting
        # the waveform on its own after this.
        self.pulse(ch=cfg["gen_ch"], t=0)
        self.sync_all()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
config = {
    "gen_ch":        GEN_CH,
    "reps":          1,
    "soft_avgs":     1,
    "frequency_mhz": frequency_mhz,
    "pulse_gain":    pulse_gain,
}

# ---------------------------------------------------------
# Run — this triggers the DAC once; the tone keeps playing
# continuously in hardware until we explicitly stop the
# generator below.
#
# NOTE: we deliberately do NOT call prog.acquire() here.
# acquire() is built around ADC polling — it computes how many
# "shots" to wait for based on declared readouts, and since this
# program declares none (it only sends), that computation blows
# up with "min() arg is an empty sequence". Since there's nothing
# to receive, we just load the program and start the tProc
# directly instead.
# ---------------------------------------------------------
prog = SendContinuousSinusoidProgram(soccfg, config)
prog.config_all(soc)
soc.tproc.start()

print(f"Continuous sinusoid running — waiting {run_time_s} s before stopping...")
time.sleep(run_time_s)

# ---------------------------------------------------------
# Stop the output. In periodic mode the DAC will keep playing
# forever otherwise (even after this script exits), so this
# step is required to actually silence the channel.
# ---------------------------------------------------------
soc.reset_gens()

print("Done — sinusoid stopped.")