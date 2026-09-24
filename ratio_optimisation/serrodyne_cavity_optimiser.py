# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:13:02 2026

@author: micha
"""

"""
Read and plot Repump_with_DDS CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
# ============================================================
# 1. Folder containing the CSV files
# ============================================================

# folder = Path(r"c:\Users\mn323\OneDrive - Imperial College London\Urop_pc_2\Serrodyne Spectra\Repump_with_DDS")
folder = Path(r"C:\Users\Ibraa\OneDrive - Imperial College London\Urop_pc_2\Serrodyne Spectra\Repump_with_DDS")

# Find all CSV files
csv_files = sorted(folder.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files found in the specified folder.")

data = {}

for file in csv_files:

    df = pd.read_csv(file, skiprows=[1])

    # Convert columns to numbers
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Channel A"] = pd.to_numeric(df["Channel A"], errors="coerce")
    df["Channel B"] = pd.to_numeric(df["Channel B"], errors="coerce")

    # Remove any rows that could not be converted
    df = df.dropna()

    data[file.name] = df


print(f"Successfully loaded {len(data)} files")


# Files to plot
files_to_plot = [
    "Repump_with_DDS_01.csv"
]

plt.figure(figsize=(10, 6))

for filename in files_to_plot:

    df = data[filename]

    plt.plot(
        df["Time"],
        df["Channel A"],
        label=filename.replace(".csv", "")
    )

plt.xlabel("Time (ms)")
plt.ylabel("Channel A (mV)")
plt.title("Repump with DDS - Channel A")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# 5. Plot ALL 64 Channel A traces

plt.figure(figsize=(12, 7))

for filename, df in data.items():

    plt.plot(
        df["Time"],
        df["Channel A"],
        alpha=0.4
    )

plt.xlabel("Time (ms)")
plt.ylabel("Channel A (mV)")
plt.title("All 64 Repump with DDS Traces - Channel A")
plt.grid(True)
plt.tight_layout()

plt.show()

for filename in files_to_plot:
    df = data[filename]

    peaks, __ = find_peaks(df["Channel A"], height=-130, prominence=20, distance=50)

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["Time"],
        df["Channel A"]
    )

    plt.plot(
        df["Time"].iloc[peaks],
        df["Channel A"].iloc[peaks],
        "x"
    )

    plt.axhline(
        -130,
        linestyle="--",
        color="gray"
    )

    peak_times = df["Time"].iloc[peaks].values
    peak_voltages = df["Channel A"].iloc[peaks].values
    
    print(f"\n{filename}")
    print("Peak indices:   ", peaks)
    print("Peak times:     ", peak_times)
    print("Peak voltages:  ", peak_voltages)
    plt.xlabel("Time")
    plt.ylabel("Channel A (V)")
    plt.title(filename)
    plt.grid(True)
    plt.show()

