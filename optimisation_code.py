# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:41:25 2026

@author: micha
"""

import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd

# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

T = 6e-3   # total sequence time

peak_voltages_main = np.array(268.85, 128.09, 97.29, 119.3)
peak_voltages_sides = np.array(20,15,12,9)
r1 = 0.25
r2 = 0.25
r3 = 0.25
r4 = 1 - sum(r1, r2, r3)
ratios = [r1, r2, r3, r4]
freqs_hz = [76.25e6, 0, -122.92e6 + 76.25e6, -147.82e6 + 76.25e6]

def calculate_error(main_peaks, side_peaks):


    mean_main = np.mean(main_peaks)

    balance_rms = (np.sqrt(np.mean((main_peaks - mean_main)**2))/ mean_main)

    sideband_rms = (np.sqrt(np.mean(side_peaks**2)) / mean_main)

    total_error = (balance_rms + sideband_rms)

    return total_error

# ============================================================
# CALCULATE CURRENT EXPERIMENTAL ERROR
# ============================================================

error = calculate_error(peak_voltages_main, peak_voltages_sides)


print("CURRENT EXPERIMENT")
print("Ratios:", ratios)
print("Main peaks:", peak_voltages_main)
print("Side peaks:", peak_voltages_sides)
print(f"\nError = {error:.6f}")
results_file = "serrodyne_optimization_results.csv"

new_result = pd.DataFrame([{
    "r1": r1,
    "r2": r2,
    "r3": r3,
    "r4": r4,

    "V1": peak_voltages_main[0],
    "V2": peak_voltages_main[1],
    "V3": peak_voltages_main[2],
    "V4": peak_voltages_main[3],

    "S1": peak_voltages_sides[0],
    "S2": peak_voltages_sides[1],
    "S3": peak_voltages_sides[2],
    "S4": peak_voltages_sides[3],

    "error": error
}])


# Append to existing CSV, or create it if it doesn't exist
try:

    existing_results = pd.read_csv(results_file)

    results = pd.concat(
        [existing_results, new_result],
        ignore_index=True
    )

except FileNotFoundError:

    results = new_result


results.to_csv(
    results_file,
    index=False
)


print("\n==============================")
print("RESULT SAVED")
print("==============================")

print(f"Saved to: {results_file}")
print(f"Total experiments recorded: {len(results)}")
