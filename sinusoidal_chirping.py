#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
"""
chirped_serrodyne_sweep.py

Single-pulse, hardware-timed linear frequency chirp for serrodyne EOM drive.

Physics / implementation notes
-------------------------------
A serrodyne (sawtooth phase-ramp) drive produces a clean single-sideband
optical frequency shift on a phase-modulating EOM because the RF phase
advances *linearly* in time and wraps every 2*pi. On this hardware that
behaviour comes "for free" from the tProc signal generator's DDS: the
generator's phase accumulator is a fixed-width register that integrates
whatever value sits in the "freq" register on every fabric-clock tick,
and it wraps automatically modulo 2*pi (i.e. modulo the register width)
-- there's no need to build or store an explicit sawtooth envelope.

To *chirp* the shift frequency (sweep it linearly in time), this program
increments the generator's "freq" register by a fixed amount every
step_time, inside a genuine tProc hardware loop (loopnz), while never
resetting the phase register in between steps. Because the frequency
register controls dphi/dt, holding it fixed for a step before bumping it
up again builds a staircase approximation to a linear frequency ramp:

    f(n)   = f_start + n * delta_f                      (n = 0 .. N-1)
    phi(n) = phi(n-1) + 2*pi*f(n)*step_time              (accumulated in HW)
           ~ integral of f(t) dt                          (N -> large limit)
           ~ phi_start + 2*pi*(f_start*t + 0.5*chirp_rate*t**2)

so the *net* phase grows quadratically in time (the "delta phi^2" the
frequency sweep produces), exactly as required for a linear-FM serrodyne
chirp. The 2*pi wrap ("keep it within range of the EOM") is handled
automatically by the finite width of the hardware phase accumulator --
we never apply an explicit modulo ourselves.

Only a handful of ASM instructions are emitted (regwi/mathi/loopnz), so
the number of chirp steps can be large (thousands) without blowing up
tProc program memory the way an unrolled Python loop, or storing every
period's envelope, would.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qick import *


class ChirpedSerrodyneProgram(AveragerProgram):
    """Plays a single linear-frequency-chirp pulse on cfg['res_ch'].

    Required cfg keys
    ------------------
    res_ch        : generator channel driving the EOM
    start_freq    : chirp start frequency [MHz]
    freq_span     : total frequency swept over the chirp [MHz]
    sweep_time    : total chirp duration [us]
    n_steps       : number of discrete frequency increments across the chirp
    pulse_gain    : DAC gain for the chirp tone [DAC units]
    relax_delay   : time to wait after the chirp before the next rep [us]
    reps          : number of tProc repetitions (hardware averaging)
    soft_avgs     : number of python-side repetitions (optional)

    Optional cfg keys
    ------------------
    ro_ch             : readout channel, if you want to trigger the ADC
                         (diagnostic only -- the readout's fixed downconversion
                         freq can't track a chirping tone, so only useful for
                         a quick loopback sanity check near the chirp start)
    readout_length    : readout window length [ADC clock ticks], required if ro_ch is set
    adc_trig_offset   : ADC trigger offset [tProc clock ticks] (default 100)
    scope_trig_pin    : output pin to pulse once at the start of the chirp,
                         for triggering an external oscilloscope (default None)
    nqz               : Nyquist zone for the generator (default 1)
    """

    # Scratch tProc register (page-local) used for the chirp step counter.
    # Register 0 is hard-wired to zero, registers 14/15 are reserved by the
    # AveragerProgram rep-loop template, and the top of the page is reserved
    # for the channel's pulse registers -- 8 is clear of all of that for a
    # single-generator program.
    CHIRP_LOOP_REG = 8

    def initialize(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg.get("nqz", 1))

        if cfg.get("ro_ch") is not None:
            self.declare_readout(
                ch=cfg["ro_ch"],
                length=cfg["readout_length"],
                freq=cfg["start_freq"],
                gen_ch=res_ch,
            )

        # --- convert user-facing (MHz / us) parameters to raw register units ---
        n_steps = int(cfg["n_steps"])
        step_time_us = cfg["sweep_time"] / n_steps
        step_length_ticks = self.us2cycles(step_time_us, gen_ch=res_ch)
        if step_length_ticks < 3:
            raise ValueError(
                f"step length of {step_length_ticks} clock ticks is too short "
                f"(minimum 3) - reduce n_steps or increase sweep_time"
            )
        if step_length_ticks >= 2 ** 16:
            raise ValueError(
                f"step length of {step_length_ticks} clock ticks exceeds the "
                f"16-bit pulse-length limit - increase n_steps"
            )

        delta_freq_MHz = cfg["freq_span"] / n_steps
        delta_freq_reg = self.freq2reg(delta_freq_MHz, gen_ch=res_ch)

        # stash the computed values for body() to use
        self.n_steps = n_steps
        self.step_length_ticks = step_length_ticks
        self.delta_freq_reg = delta_freq_reg

        print(
            f"Chirp setup: {n_steps} steps x {step_time_us:.4f} us "
            f"= {n_steps * step_time_us:.3f} us total, "
            f"{delta_freq_MHz * 1e3:.3f} kHz/step "
            f"({cfg['freq_span']:.3f} MHz span), "
            f"step length = {step_length_ticks} clock ticks"
        )

        # Base pulse definition: a plain DDS tone (no envelope needed). Its
        # frequency register is what gets incremented every chirp step in
        # body(), and its phase register is what accumulates the chirp.
        start_freq_reg = self.freq2reg(cfg["start_freq"], gen_ch=res_ch)
        self.set_pulse_registers(
            ch=res_ch,
            style="const",
            freq=start_freq_reg,
            phase=0,
            gain=cfg["pulse_gain"],
            length=step_length_ticks,
        )

        self.synci(200)  # let the tProc control core get ahead of the clock

    def body(self):
        cfg = self.cfg
        res_ch = cfg["res_ch"]
        rp = self.ch_page(res_ch)
        r_freq = self.sreg(res_ch, "freq")
        r_t = self.sreg(res_ch, "t")
        r_loop = self.CHIRP_LOOP_REG

        # Reset the DDS phase accumulator in hardware, then restore the
        # freq/phase/gain/mode registers to the values set in initialize()
        # (reset_phase() does this restoration for us). This guarantees
        # every rep starts the chirp from an identical, known phase.
        self.reset_phase(gen_ch=res_ch, t=0)

        # (re)initialize the per-channel time register and the chirp step
        # counter for this rep
        self.safe_regwi(rp, r_t, 0)
        self.regwi(rp, r_loop, self.n_steps - 1)

        # optional scope trigger marker, fired once at the start of the chirp
        pin = cfg.get("scope_trig_pin")
        if pin is not None:
            self.trigger(pins=[pin], t=0)

        # optional ADC trigger, fired once at the start of the chirp
        # (diagnostic loopback only, see class docstring)
        if cfg.get("ro_ch") is not None:
            self.trigger(
                adcs=[cfg["ro_ch"]],
                adc_trig_offset=cfg.get("adc_trig_offset", 100),
            )

        # --- the chirp itself: a genuine tProc hardware loop ---
        self.label("CHIRP_LOOP")

        # Fire the current pulse using whatever is presently in the
        # freq/phase/gain/mode/t registers (t=None => don't touch r_t here,
        # we manage it ourselves below).
        self.pulse(ch=res_ch, t=None)

        # Advance to the next step: move the per-channel time register
        # forward by one step length, and bump the frequency register by
        # delta_f. Phase is never reset here, so it keeps accumulating:
        # linear f(t) -> quadratic phi(t), wrapping mod 2*pi automatically
        # via the finite-width hardware phase accumulator.
        self.mathi(rp, r_t, r_t, "+", self.step_length_ticks)
        self.mathi(rp, r_freq, r_freq, "+", self.delta_freq_reg)

        self.loopnz(rp, r_loop, "CHIRP_LOOP")

        # Our per-step timing was managed entirely via r_t/hardware
        # registers, so QICK's Python-side timestamp bookkeeping doesn't
        # know the chirp just took n_steps*step_length_ticks clock ticks.
        # Advance the real tProc reference time by that amount...
        self.sync(rp, r_t)
        # ...and tell the Python-side bookkeeping to treat "now" as the new
        # baseline, so relax_delay below is computed correctly. (Any
        # triggered readout above finished long before the chirp ended, so
        # there's no need to wait_all() on it separately.)
        self.reset_timestamps()

        self.sync_all(self.us2cycles(cfg["relax_delay"]))


def plot_expected_chirp(cfg, filename="expected_chirp.pdf"):
    """Quick sanity-check plot of the intended frequency staircase and the
    resulting (unwrapped) phase, using pure Python -- no hardware needed.
    """
    n_steps = int(cfg["n_steps"])
    step_time_us = cfg["sweep_time"] / n_steps
    t_us = np.arange(n_steps) * step_time_us
    freq_MHz = cfg["start_freq"] + (cfg["freq_span"] / n_steps) * np.arange(n_steps)
    phase_cycles = np.cumsum(freq_MHz * step_time_us)  # unwrapped, in cycles

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(t_us, freq_MHz)
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title("Serrodyne chirp: frequency staircase")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_us, phase_cycles)
    ax2.set_xlabel("Time (us)")
    ax2.set_ylabel("Unwrapped phase (cycles)")
    ax2.set_title("Accumulated phase (quadratic growth, wraps mod 1 in HW)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    print(f"saved {filename}")


def main():
    soc = QickSoc()
    soccfg = soc

    config = {
        "res_ch": 0,             # generator channel driving the EOM
        "nqz": 1,

        "start_freq": 0.0,      # MHz, initial serrodyne shift frequency
        "freq_span": 50.0,      # MHz, total sweep span
        "sweep_time": 500000.0, #6000,   # us, total chirp duration (6 ms)
        "n_steps": 6000,         # chirp steps (1 us/step by default)
        "pulse_gain": int(np.round(32767 * 0.12)),     # DAC units

        "reps": 5, # times it repeats
        "soft_avgs": 1,
        "relax_delay": 0.0,     # us

        # remove/comment out the next three lines if you don't want an ADC
        # trigger alongside the chirp (see class docstring: diagnostic only)
        # "ro_ch": 0,
        # "readout_length": 1000,
        # "adc_trig_offset": 100,

        "scope_trig_pin": 0,
    }

    plot_expected_chirp(config)

    prog = ChirpedSerrodyneProgram(soccfg, config)
    print(prog)

    t0 = time.time()
    prog.run_rounds(soc, progress=True)
    print(f"chirp finished in {time.time() - t0:.3f} s")


if __name__ == "__main__":
    main()