# Serrodyne EOM Chirp/Trap Control for CaF Cooling & Trapping

Jupyter notebooks that drive an EOM using the **serrodyne technique** — a
phase-continuous sawtooth ramp whose slope sets an effective frequency
shift — to chirp the EOM drive across a wide frequency span (cooling/
slowing) and then hand off to a fixed multi-tone drive (trapping).

## 1. Core idea

- **Serrodyne**: a sawtooth phase ramp (`scipy.signal.sawtooth`) loaded
  into envelope memory acts as a frequency shift, so the EOM output
  frequency is set by envelope shape, not the RF synthesizer.
- **Multi-tone via time-multiplexing**: to address several CaF lines at
  once with one EOM channel, each envelope buffer concatenates short
  serrodyne segments at different frequencies (`BASE_TONES_HZ`), each
  occupying a time-share set by `TONE_RATIOS`. Played in
  `mode="periodic"`, the EOM sees a frequency comb whose spacing is set by
  the buffer's repeat rate and whose line power is set by each segment's
  ratio/amplitude.
- **Chirping**: the sweep span is broken into `NUM_STEPS` steps, each a
  different center frequency, advanced entirely in tProc hardware (no
  Python unrolling) once per `STEP_HOLD_US`.
- **Trapping**: after the sweep, one more envelope is played periodically
  at a fixed `TRAP_OFFSET_HZ` and left running until `soc.reset_gens()`.

## 2. Program

* `Optimised_tunability_Aamtrscste.ipynb` — sweep goes `CHIRP_OFFSET_START_HZ = -230 MHz → CHIRP_OFFSET_STOP_HZ = +120 MHz`, while `TRAP_OFFSET_HZ = 0 MHz` — i.e. the sweep doesn't end exactly at the trap frequency. `TONE_RATIOS`/`AMPLITUDE` are per-step arrays (`TONE_RATIOS_PER_STEP`, `AMPLITUDE_PER_STEP`), so each tone's weighting can vary across the sweep. Because only phase (not frequency) is carried continuously into the trap buffer, this produces a deliberate, instantaneous ~120 MHz frequency jump at the sweep→trap handoff (still phase-continuous, just not frequency-continuous) — accepted so the sweep can reach +120 MHz before switching to the 0 MHz trap tone. +120 MHz is the Doppler shift of the beam at the handoff/capture velocity, while 0 MHz (resonance) is right for a molecule that's essentially at rest in the trap. `best_n_samples()` picks each tone's sample count from a small grid-aligned window around the target duration to minimize residual phase at the periodic-buffer wraparound, ties broken by proximity to the target rather than list order. Uses midpoint chirp sampling (steps evaluated at their dwell-interval midpoint, not the start, with the last step snapped exactly to the sweep endpoint) to cut staircase phase error 4x. Includes an automatic shrink-to-fit loop for envelope memory (shrinking `CYCLE_S` and rebuilding, without affecting sweep timing, up to 60 attempts, raising if any tone slice would drop below 3 fabric cycles), plus a measured EOM diffraction-efficiency calibration: each tone's drive amplitude is corrected by `get_correction_multiplier(freq)` (interpolated from measured data, extrapolated below −50 MHz, folded symmetrically onto positive offsets) so every tone gets equal diffracted power regardless of frequency, with `BASE_AMPLITUDE` auto-scaled down if that correction would exceed full DAC scale. Runs as `RepeatedStepSerrodyneProgram` (`RAveragerProgram`), stepping through per-step envelopes entirely in tProc hardware via register-address increment, then appending a single periodic trap pulse after the loop that the DAC keeps cycling indefinitely until `soc.reset_gens()`, with external-trigger start (`start_src="external"`).

## 3. Key parameters (common across notebooks)

| Parameter | Meaning |
|---|---|
| `GEN_CH` / `RO_CH` | QICK DAC generator / ADC readout channel driving the EOM |
| `BASE_TONES_HZ` | Frequency offsets of the simultaneous tones (e.g. main + sideband lines for CaF transitions), relative to the chirp center |
| `TONE_RATIOS` / `TONE_RATIOS_PER_STEP` | Relative time-share of each tone within a composite buffer (sets relative line power, along with amplitude) |
| `AMPLITUDE` / `AMPLITUDE_PER_STEP` / `BASE_AMPLITUDE` | Serrodyne drive amplitude (0–1 of full DAC scale) per tone |
| `CHIRP_OFFSET_START_HZ` / `CHIRP_OFFSET_STOP_HZ` | Start/stop of the frequency sweep |
| `TRAP_OFFSET_HZ` | Frequency to hold at after the sweep finishes |
| `NUM_STEPS` | Number of discrete frequency steps in the chirp |
| `TOTAL_SWEEP_S` | Physical wall-clock duration of the whole sweep |
| `CYCLE_S` | Duration of one composite buffer's repeat period (auto-shrunk to fit memory; does not affect `TOTAL_SWEEP_S`) |
| `ENV_MAXLEN` | Envelope memory available on the chosen generator channel (samples) |

## 4. Running from Python

Before launching Jupyter/Python on the board, set up the environment:

```
sudo su
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
```
