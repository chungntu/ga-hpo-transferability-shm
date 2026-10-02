"""
Convert the raw QUGS ME'scope text exports into per-record .npy arrays.

Replaces QUGS/getData_v2.m + getData_v3.m + chiaFile.m, but WITHOUT the
splitting step: splitting a single continuous record into chunks and then
spreading those chunks over train/val/test is record-level leakage.
Windowing is done later, at
training time, inside the record-level split.

Layout produced:
    revision/data/qugs/A/state01.npy ... state31.npy   (262144, 30) float32
    revision/data/qugs/B/state01.npy ... state31.npy

state01 = undamaged reference (zzzAU / zzzBU)
state02..state31 = loosened bolt at joint 1..30 (zzzAD1..AD30 / zzzBD1..BD30)

Sampling rate 1024 Hz, 256 s per record, 30 accelerometers.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))

import os
import re
import sys
import time
import zipfile

import numpy as np

RAW_DIR = r"C:\09 VERY BIG DATA\QUGS"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "qugs")

HEADER_LINES = 11      # 11 metadata lines before the first numeric row
N_SENSORS = 30         # columns 2..31 (column 1 is time)
N_ROWS = 262144


def record_name_to_state(stem):
    """zzzAU -> 1 ; zzzAD7 -> 8 ; zzzBD30 -> 31"""
    m = re.fullmatch(r"zzz[AB]U", stem)
    if m:
        return 1
    m = re.fullmatch(r"zzz[AB]D(\d+)", stem)
    if m:
        return int(m.group(1)) + 1
    raise ValueError(f"unrecognised record name: {stem}")


def parse_stream(fh):
    """Read the numeric block into a (N_ROWS, N_SENSORS) float32 array."""
    for _ in range(HEADER_LINES):
        fh.readline()
    rows = []
    for line in fh:
        parts = line.split(b"\t")
        if len(parts) < N_SENSORS + 1:
            continue
        rows.append(parts[1:N_SENSORS + 1])
    arr = np.array(rows, dtype=np.bytes_).astype(np.float32)
    return arr


def main():
    zips = [f for f in sorted(os.listdir(RAW_DIR)) if f.lower().endswith(".zip")]
    if not zips:
        sys.exit(f"no zip archives under {RAW_DIR}")

    todo = []
    for zname in zips:
        zp = os.path.join(RAW_DIR, zname)
        with zipfile.ZipFile(zp) as z:
            for entry in z.namelist():
                if not entry.upper().endswith(".TXT"):
                    continue
                top = entry.split("/")[0]                 # "Dataset A" / "Dataset B"
                subset = top.rsplit(" ", 1)[-1]           # "A" / "B"
                stem = os.path.splitext(os.path.basename(entry))[0]
                state = record_name_to_state(stem)
                todo.append((zp, entry, subset, state, stem))

    print(f"[prepare_qugs] {len(todo)} records found across {len(zips)} archives", flush=True)

    for i, (zp, entry, subset, state, stem) in enumerate(todo, 1):
        out_sub = os.path.join(OUT_DIR, subset)
        os.makedirs(out_sub, exist_ok=True)
        out_path = os.path.join(out_sub, f"state{state:02d}.npy")
        if os.path.exists(out_path):
            print(f"[{i}/{len(todo)}] skip {subset}/state{state:02d} (exists)", flush=True)
            continue

        t0 = time.perf_counter()
        with zipfile.ZipFile(zp) as z, z.open(entry) as fh:
            arr = parse_stream(fh)

        if arr.shape[0] != N_ROWS or arr.shape[1] != N_SENSORS:
            print(f"[WARN] {stem}: unexpected shape {arr.shape}", flush=True)

        tmp = out_path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_path)
        print(
            f"[{i}/{len(todo)}] {subset}/state{state:02d} <- {stem} "
            f"{arr.shape} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    print("[prepare_qugs] done", flush=True)


if __name__ == "__main__":
    main()
