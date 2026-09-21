import sys
sys.path.insert(0, "/home/xilinx/jupyter_notebooks/qick/qick_lib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qick import *
from scipy import signal as scipy_signal
from tqdm import tqdm

soc    = QickSoc()
soccfg = soc

GEN_CH = 1
RO_CH  = 0   # kept only because the tProc must be run via acquire_decimated()

gencfg        = soccfg["gens"][GEN_CH]
samps_per_clk = gencfg["samps_per_clk"]
ENV_SR        = gencfg["f_fabric"] * samps_per_clk * 1e6
MAXLEN        = gencfg["maxlen"]

print(f"ENV_SR={ENV_SR/1e9:.4f} GSPS, samps_per_clk={samps_per_clk}, maxlen={MAXLEN}")

# ── chirp parameters ──────────────────────────────────────────────────────────
base_freqs_hz   = np.array([-76.25e6, 0.0, 122.92e6 - 76.25e6, 147.82e6 - 76.25e6])
base_ratios     = [0.337, 0.167, 0.288, 0.208]
chirp_offset    = 350e6
amp             = 1.0
width           = 1.0
total_chirp_s   = 6e-3          # 6 ms ← the real target

steps           = 5
sweep_cfg       = {"start": 0e6, "step": chirp_offset / steps, "expts": steps}
frequency_sweep_hz = sweep_cfg["start"] + sweep_cfg["step"] * np.arange(sweep_cfg["expts"])

# ── segment builder ───────────────────────────────────────────────────────────
def min_seg_len_for_freqs(ratios, freqs_hz, sample_rate, samps_per_clk):
    """Smallest total segment length (multiple of samps_per_clk) such that
    each non-zero sub-segment contains >= 1 full period."""
    ratio_sum = sum(ratios)
    min_total = samps_per_clk
    for ratio, freq in zip(ratios, freqs_hz):
        if abs(freq) < 1e3:
            continue
        one_period   = sample_rate / abs(freq)
        needed_total = one_period * ratio_sum / ratio
        needed_total = int(np.ceil(needed_total / samps_per_clk)) * samps_per_clk
        min_total    = max(min_total, needed_total)
    return min_total


def build_segment(ratios, freqs_hz, *, amplitude, sample_rate,
                  samps_per_clk, seg_len, phase_offset=0.0, width=1.0):
    """Build one composite serrodyne segment of exactly seg_len samples."""
    two_pi    = 2.0 * np.pi
    dt        = 1.0 / sample_rate
    ratio_sum = sum(ratios)

    sub_lengths = []
    for ratio in ratios:
        n = int(round(seg_len * ratio / ratio_sum / samps_per_clk)) * samps_per_clk
        n = max(samps_per_clk, n)
        sub_lengths.append(n)

    # Fix rounding so total == seg_len
    diff = seg_len - sum(sub_lengths)
    idx  = np.argsort(sub_lengths)[::-1]
    i    = 0
    while diff != 0:
        step      = samps_per_clk if diff > 0 else -samps_per_clk
        candidate = sub_lengths[idx[i % len(idx)]] + step
        if candidate >= samps_per_clk:
            sub_lengths[idx[i % len(idx)]] = candidate
            diff -= step
        i += 1
        if i > len(sub_lengths) * 100:
            break

    y         = []
    phase_out = phase_offset

    for length, freq in zip(sub_lengths, freqs_hz):
        if abs(freq) < 1e3:          # DC
            y.append(np.zeros(length))
        else:
            t   = np.arange(length) * dt
            phi = two_pi * freq * t + phase_out
            y.append(amplitude * scipy_signal.sawtooth(phi, width=width))
            phase_out = (two_pi * freq * (length * dt) + phase_out) % two_pi

    y_out = np.concatenate(y)
    pad   = (-len(y_out)) % samps_per_clk
    if pad:
        y_out = np.concatenate([y_out, np.zeros(pad)])

    maxv  = soccfg.get_maxv(GEN_CH)
    idata = np.round(y_out * maxv).astype(np.int16)
    qdata = np.zeros_like(idata)
    return idata, qdata, len(y_out), phase_out


def build_packed_envelope(base_ratios, freq_start, freq_end,
                          chirped_number, sample_rate, samps_per_clk, maxlen):
    """
    Build the full packed envelope for one sweep point.
    Returns packed_i, packed_q, seg_len_samples, seg_len_words, chirped_number_actual
    """
    # Worst-case segment length (lowest frequencies = largest magnitudes)
    worst_freqs    = freq_start   # largest magnitudes at start
    nat_seg_len    = min_seg_len_for_freqs(base_ratios, worst_freqs,
                                           sample_rate, samps_per_clk)
    max_n_steps    = maxlen // nat_seg_len
    chirped_number = min(chirped_number, max_n_steps)

    if chirped_number < 2:
        raise RuntimeError(
            f"Natural segment length {nat_seg_len} too large for maxlen {maxlen}. "
            f"Frequencies are too low for this memory budget."
        )

    seg_len_samples = nat_seg_len
    seg_len_words   = seg_len_samples // samps_per_clk

    # Build chirp frequency table
    freq_table = np.array([
        np.linspace(freq_start[i], freq_end[i], chirped_number)
        for i in range(len(base_ratios))
    ]).T   # (chirped_number, n_components)

    all_idata = []
    all_qdata = []
    phase_acc = 0.0

    for i in range(chirped_number):
        idata, qdata, n, phase_acc = build_segment(
            base_ratios, freq_table[i],
            amplitude=1.0,
            sample_rate=sample_rate,
            samps_per_clk=samps_per_clk,
            seg_len=seg_len_samples,
            phase_offset=phase_acc,
            width=1.0,
        )
        all_idata.append(idata)
        all_qdata.append(qdata)

    packed_i = np.zeros(chirped_number * seg_len_samples, dtype=np.int16)
    packed_q = np.zeros(chirped_number * seg_len_samples, dtype=np.int16)

    for i, (id_, qd_) in enumerate(zip(all_idata, all_qdata)):
        s = i * seg_len_samples
        n = min(len(id_), seg_len_samples)
        packed_i[s:s+n] = id_[:n]
        packed_q[s:s+n] = qd_[:n]

    return packed_i, packed_q, seg_len_samples, seg_len_words, chirped_number


# ── tProc program ─────────────────────────────────────────────────────────────
class ChirpProgram(AveragerProgram):
    """
    Plays a 6 ms serrodyne chirp via addr-stepping (mode='periodic').
    No meaningful ADC capture — the readout below is only present because
    the tProc has to be started through acquire_decimated().

    cfg keys
    --------
    gen_ch          : int
    ro_chs          : list[int]
    packed_idata    : np.ndarray int16
    packed_qdata    : np.ndarray int16
    seg_len_words   : int
    chirped_number  : int
    dwell_us        : float      µs per chirp step
    gain            : int
    relax_delay     : float      µs after the chirp before next rep
    """

    def initialize(self):
        cfg = self.cfg
        ch  = cfg["gen_ch"]

        self.declare_gen(ch=ch, nqz=1)

        # Minimal placeholder readout — required only to run the tProc.
        for ro_ch in cfg["ro_chs"]:
            self.declare_readout(ch=ro_ch, length=100, freq=0, gen_ch=ch)

        self.add_envelope(
            ch=ch,
            name="chirp_table",
            idata=cfg["packed_idata"],
            qdata=cfg["packed_qdata"],
        )

        self.set_pulse_registers(
            ch=ch,
            style="arb",
            freq=0,
            phase=0,
            gain=cfg["gain"],
            mode="periodic",
            outsel="input",
            phrst=0,
            waveform="chirp_table",   # addr starts at 0
        )

        self.r_page = self.ch_page(ch)
        self.r_addr = self.sreg(ch, "addr")
        self.r_loop = 1   # free register for loop counter

        # Initialise loop counter
        self.regwi(self.r_page, self.r_loop,
                   cfg["chirped_number"] - 1, "loop counter")

        self.synci(200)

    def body(self):
        cfg          = self.cfg
        ch           = cfg["gen_ch"]
        dwell_cycles = self.us2cycles(cfg["dwell_us"])

        # ── chirp loop ────────────────────────────────────────────────────
        self.label("CHIRP_LOOP")

        self.pulse(ch=ch, t="auto")
        self.sync_all(dwell_cycles)

        self.mathi(
            self.r_page,
            self.r_addr,
            self.r_addr,
            "+",
            cfg["seg_len_words"],
        )

        self.loopnz(self.r_page, self.r_loop, "CHIRP_LOOP")

        # ── done — silence and relax ──────────────────────────────────────
        self.sync_all(self.us2cycles(cfg["relax_delay"]))

    def run(self, soc, **kwargs):
        """Play the chirp on the DAC. ADC data is discarded/unused."""
        return super().acquire_decimated(soc, **kwargs)


# ── main sweep loop ───────────────────────────────────────────────────────────
maxv = soccfg.get_maxv(GEN_CH)
first_point_envelope = None   # saved for plotting

for sweep_idx, common_offset in enumerate(tqdm(frequency_sweep_hz)):

    # ── frequencies for this sweep point ─────────────────────────────────
    current_freqs_hz = base_freqs_hz + common_offset      # end of chirp
    freq_hz_shifted  = current_freqs_hz + chirp_offset    # start of chirp

    # Sign flip for serrodyne convention
    freq_start = -freq_hz_shifted
    freq_end   = -current_freqs_hz

    # ── build packed envelope ─────────────────────────────────────────────
    chirped_number = 100   # will be clamped to what fits in maxlen
    packed_i, packed_q, seg_len_samples, seg_len_words, chirped_number = \
        build_packed_envelope(
            base_ratios, freq_start, freq_end,
            chirped_number,
            ENV_SR, samps_per_clk, MAXLEN,
        )

    dwell_us = (total_chirp_s / chirped_number) * 1e6

    if sweep_idx == 0:
        seg_dur_us = seg_len_samples / ENV_SR * 1e6
        print(f"\n=== Envelope plan (sweep point 0) ===")
        print(f"  chirped_number   : {chirped_number}")
        print(f"  seg_len_samples  : {seg_len_samples}")
        print(f"  seg_len_words    : {seg_len_words}")
        print(f"  seg duration     : {seg_dur_us:.4f} µs")
        print(f"  dwell_us         : {dwell_us:.3f} µs")
        print(f"  repeats/dwell    : {dwell_us/seg_dur_us:.1f}×")
        print(f"  total buffer     : {len(packed_i)} / {MAXLEN} samples "
              f"({100*len(packed_i)/MAXLEN:.1f}%)")
        print(f"  total chirp time : {chirped_number*dwell_us/1e3:.3f} ms")
        first_point_envelope = (packed_i.copy(), dwell_us, seg_len_samples, chirped_number)

    # ── program config ────────────────────────────────────────────────────
    prog_cfg = {
        "gen_ch":         GEN_CH,
        "ro_chs":         [RO_CH],
        "packed_idata":   packed_i,
        "packed_qdata":   packed_q,
        "seg_len_words":  seg_len_words,
        "chirped_number": chirped_number,
        "dwell_us":       dwell_us,
        "gain":           1000,
        "relax_delay":    3,
        "reps":           1,
    }

    prog = ChirpProgram(soccfg, prog_cfg)
    prog.run(soc, progress=False)

# ── plot the commanded DAC envelope (sweep point 0) ─────────────────────────
packed_i0, dwell_us0, seg_len_samples0, chirped_number0 = first_point_envelope
t_us = np.arange(len(packed_i0)) / ENV_SR * 1e6

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(t_us, packed_i0, lw=0.5)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("DAC I (int16)")
ax.set_title("Commanded DAC Envelope — Sweep Point 0")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/home/xilinx/jupyter_notebooks/qick/qick_demos/pictures/"
    "doppler_cooling_output_signals.pdf",
    bbox_inches="tight",
)
plt.close()
print("Done.")
