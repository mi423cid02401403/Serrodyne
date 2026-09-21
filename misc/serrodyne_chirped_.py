"""
serrodyne_demo.py  —  fixed for RFSoC4x2, QICK v0.2.422
"""

import numpy as np
import matplotlib.pyplot as plt
from qick import QickSoc
from qick.averager_program import AveragerProgram

# ─── HARDWARE CONSTANTS (read from your soccfg printout) ─────────────────────
# tProc clock  : 409.600 MHz
# DAC fabric   : 614.400 MHz
# ADC decimated: 552.960 MHz

# ─── CONFIG ──────────────────────────────────────────────────────────────────
config = {
    # channels
    "gen_ch"          : 0,        # DAC channel 0  (axis_signal_gen_v6)
    "ro_ch"           : 0,        # ADC channel 0  (axis_readout_v2)

    # pulse
    "f0_MHz"          : 0.0,    # base DDS frequency in MHz
    "delta_f_MHz"     : 350.0,      # serrodyne shift in MHz
    "gain"            : 5000,     # DAC gain (max 32767)

    # serrodyne timing
    # quantum = 3 fabric cycles per loop iteration
    # at 614.4 MHz fabric, 3 cycles = ~4.88 ns per step
    # keep n_steps small enough: n_steps * 3 < 65536
    "n_steps"         : 300,     # number of sawtooth steps  (<= 21845)

    # readout
    "readout_length"  : 1020,     # ADC decimated samples
    "adc_trig_offset" : 270,      # tProc cycles

    # averager
    "reps"            : 10,
    "relax_delay_us"  : 0.0,
}

# ─── PROGRAM ─────────────────────────────────────────────────────────────────

class SerrodyneProgram(AveragerProgram):
    """
    Serrodyne frequency shift using tProc phase-ramp loop.

    Inherits from AveragerProgram so QICK handles the rep loop,
    shot counting, and acquisition automatically.

    The assembly body:
        1. Writes a short periodic DDS pulse (3 fabric cycles)
        2. Increments the phase register by phase_step
        3. Loops n_steps times  →  sawtooth phase ramp
        4. Triggers ADC
    """

    def initialize(self):
        cfg = self.cfg

        # ── validate n_steps ──────────────────────────────────────────────
        quantum   = 3                          # min legal pulse length (fabric cycles)
        n_steps   = cfg["n_steps"]
        total_len = n_steps * quantum

        if total_len >= 2**16:
            raise ValueError(
                f"total_len = n_steps({n_steps}) * quantum({quantum}) = {total_len} "
                f"exceeds 16-bit limit (65535). Reduce n_steps to <= {65535 // quantum}."
            )

        self.quantum   = quantum
        self.n_steps   = n_steps
        self.total_len = total_len

        # ── declare readout ───────────────────────────────────────────────
        self.declare_readout(
            ch     = cfg["ro_ch"],
            length = cfg["readout_length"],
            freq   = cfg["f0_MHz"],
            gen_ch = cfg["gen_ch"],
        )

        # ── convert frequencies ───────────────────────────────────────────
        self.f0_reg = self.freq2reg(
            cfg["f0_MHz"],
            gen_ch = cfg["gen_ch"],
            ro_ch  = cfg["ro_ch"],
        )

        # ── compute serrodyne phase step ──────────────────────────────────
        # delta_f = phase_step * f_tproc / 2^32
        # phase_step = delta_f * 2^32 / f_tproc
        f_tproc_MHz    = self.tproccfg["f_time"]   # 409.6 MHz
        self.phase_step = int(round(
            cfg["delta_f_MHz"] * (2**32) / f_tproc_MHz
        ))

        print(f"[SerrodyneProgram] f_tproc      = {f_tproc_MHz} MHz")
        print(f"[SerrodyneProgram] phase_step   = {self.phase_step}")
        print(f"[SerrodyneProgram] n_steps      = {self.n_steps}")
        print(f"[SerrodyneProgram] total_len    = {self.total_len} fab cycles")
        print(f"[SerrodyneProgram] actual delta_f = "
              f"{self.phase_step * f_tproc_MHz / 2**32:.6f} MHz")

        # ── default pulse registers (written once here, not every rep) ────
        self.default_pulse_registers(
            cfg["gen_ch"],
            freq  = self.f0_reg,
            gain  = cfg["gain"],
            phase = 0,
        )

        # ── readout registers ─────────────────────────────────────────────
        self.set_readout_registers(
            cfg["ro_ch"],
            freq   = self.freq2reg_adc(cfg["f0_MHz"], cfg["ro_ch"]),
            length = cfg["readout_length"],
            mode   = "oneshot",
            outsel = "product",
        )

        self.synci(200)   # settling time

    def body(self):
        cfg = self.cfg

        # ── program the serrodyne pulse ───────────────────────────────────
        self.set_pulse_registers(
            cfg["gen_ch"],
            style      = "serrodyne",
            length     = self.total_len,
            phase_step = self.phase_step,
            n_steps    = self.n_steps,
        )

        # ── trigger ADC ───────────────────────────────────────────────────
        self.trigger(
            adcs            = [cfg["ro_ch"]],
            adc_trig_offset = cfg["adc_trig_offset"],
        )

        # ── play serrodyne pulse ──────────────────────────────────────────
        self.pulse(
            ch = cfg["gen_ch"],
            t  = "auto",
        )

        # ── wait for readout window to close ──────────────────────────────
        self.wait_all()

        # ── relax before next rep ─────────────────────────────────────────
        self.sync_all(
            self.us2cycles(cfg["relax_delay_us"])
        )


# ─── FREQUENCY SWEEP PROGRAM ──────────────────────────────────────────────────

class SerrodyneSweeProgram(AveragerProgram):
    """
    Sweeps delta_f across cfg["delta_f_list_MHz"].
    Each rep plays all delta_f values sequentially.

    cfg must include:
        "delta_f_list_MHz" : list of float
    """

    def initialize(self):
        cfg = self.cfg

        quantum = 3
        n_steps = cfg["n_steps"]
        total_len = n_steps * quantum

        if total_len >= 2**16:
            raise ValueError(
                f"total_len={total_len} exceeds 16-bit limit. "
                f"Reduce n_steps to <= {65535 // quantum}."
            )

        self.quantum   = quantum
        self.n_steps   = n_steps
        self.total_len = total_len

        self.declare_readout(
            ch     = cfg["ro_ch"],
            length = cfg["readout_length"],
            freq   = cfg["f0_MHz"],
            gen_ch = cfg["gen_ch"],
        )

        self.f0_reg = self.freq2reg(
            cfg["f0_MHz"],
            gen_ch = cfg["gen_ch"],
            ro_ch  = cfg["ro_ch"],
        )

        f_tproc_MHz = self.tproccfg["f_time"]

        # precompute all phase steps
        self.phase_steps = [
            int(round(df * (2**32) / f_tproc_MHz))
            for df in cfg["delta_f_list_MHz"]
        ]

        self.default_pulse_registers(
            cfg["gen_ch"],
            freq  = self.f0_reg,
            gain  = cfg["gain"],
            phase = 0,
        )

        self.set_readout_registers(
            cfg["ro_ch"],
            freq   = self.freq2reg_adc(cfg["f0_MHz"], cfg["ro_ch"]),
            length = cfg["readout_length"],
            mode   = "oneshot",
            outsel = "product",
        )

        self.synci(200)

    def body(self):
        cfg = self.cfg

        for phase_step in self.phase_steps:

            # reset phase before each sub-pulse
            self.reset_phase(gen_ch=cfg["gen_ch"], t=0)

            self.set_pulse_registers(
                cfg["gen_ch"],
                style      = "serrodyne",
                length     = self.total_len,
                phase_step = phase_step,
                n_steps    = self.n_steps,
            )

            self.trigger(
                adcs            = [cfg["ro_ch"]],
                adc_trig_offset = cfg["adc_trig_offset"],
            )

            self.pulse(ch=cfg["gen_ch"], t="auto")
            self.wait_all()
            self.sync_all(self.us2cycles(cfg["relax_delay_us"]))


# ─── RUNNERS ─────────────────────────────────────────────────────────────────

def run_serrodyne(soc, soccfg, cfg):
    """
    Run single serrodyne pulse and return averaged I/Q.

    Returns
    -------
    I : np.ndarray  shape (readout_length,)
    Q : np.ndarray  shape (readout_length,)
    """
    prog = SerrodyneProgram(soccfg, cfg)

    print("\n" + "="*60)
    print("ASSEMBLY")
    print("="*60)
    print(prog.asm())
    print("="*60 + "\n")

    I, Q = prog.acquire(
        soc,
        reads_per_rep = 1,
        load_pulses   = True,
        progress      = True,
    )

    # acquire returns shape (n_ro, reps, readout_length) → take first ro, average reps
    return I[0][0], Q[0][0]


def run_serrodyne_sweep(soc, soccfg, cfg, delta_f_list_MHz):
    """
    Sweep delta_f and return I/Q for each value.

    Returns
    -------
    I_list : np.ndarray  shape (n_freqs, readout_length)
    Q_list : np.ndarray  shape (n_freqs, readout_length)
    """
    sweep_cfg = {**cfg, "delta_f_list_MHz": delta_f_list_MHz}
    prog = SerrodyneSweeProgram(soccfg, sweep_cfg)

    print(prog.asm())

    I, Q = prog.acquire(
        soc,
        reads_per_rep = len(delta_f_list_MHz),
        load_pulses   = True,
        progress      = True,
    )

    return I[:, 0, :], Q[:, 0, :]


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_iq_time(I, Q, cfg, save=True):
    t = np.arange(len(I))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, I, color="royalblue", label="I")
    axes[0].set_ylabel("I (ADC units)")
    axes[0].legend(); axes[0].grid(True)
    axes[0].set_title(
        f"Serrodyne  |  f0={cfg['f0_MHz']} MHz  "
        f"Δf={cfg['delta_f_MHz']} MHz  "
        f"gain={cfg['gain']}  n_steps={cfg['n_steps']}"
    )

    axes[1].plot(t, Q, color="tomato", label="Q")
    axes[1].set_ylabel("Q (ADC units)")
    axes[1].set_xlabel("ADC sample index")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    if save:
        plt.savefig("serrodyne_iq_time.png", dpi=150)
        print("Saved: serrodyne_iq_time.png")
    plt.show()


def plot_iq_sweep(I_list, Q_list, delta_f_list_MHz, save=True):
    I_mean = np.array([np.mean(i) for i in I_list])
    Q_mean = np.array([np.mean(q) for q in Q_list])
    amp    = np.sqrt(I_mean**2 + Q_mean**2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(delta_f_list_MHz, I_mean, "o-", color="royalblue")
    axes[0].set_ylabel("Mean I"); axes[0].grid(True)
    axes[0].set_title("Serrodyne Sweep")

    axes[1].plot(delta_f_list_MHz, Q_mean, "o-", color="tomato")
    axes[1].set_ylabel("Mean Q"); axes[1].grid(True)

    axes[2].plot(delta_f_list_MHz, amp, "o-", color="seagreen")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Δf (MHz)"); axes[2].grid(True)

    plt.tight_layout()
    if save:
        plt.savefig("serrodyne_sweep.png", dpi=150)
        print("Saved: serrodyne_sweep.png")
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── connect ───────────────────────────────────────────────────────────
    soc    = QickSoc()
    soccfg = soc
    print(soccfg)

    # ── sanity check before touching hardware ─────────────────────────────
    print("\n--- Dry-run sanity check ---")
    prog = SerrodyneProgram(soccfg, config)
    print(prog.asm())

    # ── single pulse ──────────────────────────────────────────────────────
    I, Q = run_serrodyne(soc, soccfg, config)

    print(f"\nMean I    = {np.mean(I):.2f}")
    print(f"Mean Q    = {np.mean(Q):.2f}")
    print(f"Amplitude = {np.mean(np.sqrt(I**2 + Q**2)):.2f}")

    plot_iq_time(I, Q, config)

    # ── frequency sweep ───────────────────────────────────────────────────
    delta_f_list = np.linspace(-5.0, 5.0, 21).tolist()

    I_list, Q_list = run_serrodyne_sweep(
        soc, soccfg, config, delta_f_list
    )

    plot_iq_sweep(I_list, Q_list, delta_f_list) 
