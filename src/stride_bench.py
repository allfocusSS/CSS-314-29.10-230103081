#!/usr/bin/env python3
"""Milestone 1, Task 1.1: stride latency cliff benchmark.

A flat float64 buffer of 2^24 elements (~134 MB, larger than L3) is reduced with
stride s in {1,2,4,...,128}.  The reduction is done with NumPy on a strided view
(data[::s].sum()): a pure-Python `for i in range(0, n, s)` loop would be dominated
by interpreter overhead (~50 ns/iteration) and hide the memory effect we want to see.

5 warm-up iterations, then 5 recorded iterations; the median time_ns is used.
Output: results/stride_results.tsv
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
WARMUP, REPS = 5, 5


def bench_stride(data, s):
    view = data[::s]
    touched = view.shape[0]
    total = 0.0
    for _ in range(WARMUP):
        total = view.sum()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        total = view.sum()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return touched, times, float(total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--elements", type=int, default=16_777_216,
                    help="buffer size in float64 elements (default 2^24; keep it > L3 for the real run)")
    args = ap.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Allocate flat array in physical memory
    data = np.ones(args.elements, dtype=np.float64)
    print(f"buffer: {args.elements} float64 = {data.nbytes/1e6:.1f} MB")

    rows, base = [], None
    for s in STRIDES:
        touched, times, total = bench_stride(data, s)
        med = int(statistics.median(times))
        lat = med / touched
        base = base or lat
        rows.append((s, 8 * s, touched, med, lat, lat / base, times, total))
        print(f"stride {s:4d} ({8*s:5d} B): {touched:9d} elems  median {med/1e6:9.3f} ms  "
              f"{lat:7.3f} ns/elem  x{lat/base:5.2f} vs stride-1")

    with open(out / "stride_results.tsv", "w") as f:
        f.write("stride\tstride_bytes\telements_touched\tmedian_time_ns\tlatency_ns_per_element\t"
                "relative_to_stride1\trun_times_ns\tchecksum\n")
        for s, sb, touched, med, lat, rel, times, total in rows:
            f.write(f"{s}\t{sb}\t{touched}\t{med}\t{lat:.4f}\t{rel:.3f}\t"
                    f"{','.join(map(str, times))}\t{total:.1f}\n")
    print(f"wrote {out/'stride_results.tsv'}")

    lat1 = rows[0][4]
    lat8 = rows[3][4]
    if lat8 / lat1 < 1.5:
        print("WARNING: stride-8 is <1.5x slower per element than stride-1 - the cliff is weak. "
              "Close other programs, check the buffer is larger than your L3, and re-run.",
              file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
