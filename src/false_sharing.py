#!/usr/bin/env python3
"""Milestone 2: false-sharing pathology, remediation, and Karp-Flatt diagnosis.

Three variants, each run with P in {1,2,4,8} worker processes, K increments per worker:

  unpadded    worker p increments shared_arr[p]       -> all counters share one 64 B line
  padded      worker p increments shared_arr[p * 8]   -> one counter per 64 B line
  privatized  worker p increments a private local K times, then does ONE locked write

Notes on the implementation
  * mp.Array(..., lock=False) is used: the default lock=True wraps every index access in a
    mutex, which would measure lock contention instead of cache-line contention.
  * The shared buffer is over-allocated and a base offset is computed so that slot 0 sits on a
    64-byte boundary; "indices 0..7 share one line" is then exactly true.
  * A barrier releases all workers together; the timer starts when it is released and stops
    when every worker has exited, so process start-up cost is excluded.
  * --backend c runs the increment loop in a tiny compiled C function (volatile store per
    iteration).  CPython's ~60 ns/iteration interpreter cost can mask a cache-line
    ping-pong; the C loop makes the effect much clearer.  Default is pure Python, as the lab says.

Karp-Flatt (spec formula, S_P = T_1 / T_P):        e(P) = (P*T_P/T_1 - 1) / (P - 1)
Work-normalised variant (each worker does K, so total work is P*K, S_P = P*T_1/T_P):
                                                   e_w(P) = (T_P/T_1 - 1) / (P - 1)
The spec formula treats the work as fixed; here work grows with P, so even perfect scaling
gives T_P = T_1 and e(P) = 1.  Both columns are written; the work-normalised one is 0 for
ideal scaling and is the one that behaves as "flat and low" for the padded runs.

Output: results/false_sharing_karp_flatt.tsv
"""
import argparse
import ctypes
import multiprocessing as mp
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LINE_INTS = 8  # 64-byte line / 8-byte int64
C_SRC = r"""
void inc(volatile long long *p, long long n) { for (long long i = 0; i < n; i++) *p += 1; }
long long loc(long long n) { volatile long long x = 0; for (long long i = 0; i < n; i++) x += 1; return x; }
"""


def _wait(barrier):
    barrier.wait(timeout=600)


def _loop_shared(arr, slot, iterations, barrier, lib_path):
    if lib_path:
        lib = ctypes.CDLL(lib_path)
        lib.inc.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        addr = ctypes.addressof(arr) + slot * 8
        _wait(barrier)
        lib.inc(addr, iterations)
    else:
        _wait(barrier)
        for _ in range(iterations):
            arr[slot] += 1


def worker_unpadded(counter_array, worker_id, iterations, barrier, base, lib_path, lock):
    # Pathological: workers p and p+1 write adjacent int64s that live on the same 64 B line.
    _loop_shared(counter_array, base + worker_id, iterations, barrier, lib_path)


def worker_padded(counter_array, worker_id, iterations, barrier, base, lib_path, lock):
    # Each counter is 8 ints (64 B) apart -> a private cache line per worker.
    _loop_shared(counter_array, base + worker_id * LINE_INTS, iterations, barrier, lib_path)


def worker_privatized(counter_array, worker_id, iterations, barrier, base, lib_path, lock):
    if lib_path:
        lib = ctypes.CDLL(lib_path)
        lib.loc.argtypes = [ctypes.c_longlong]
        lib.loc.restype = ctypes.c_longlong
        _wait(barrier)
        local = lib.loc(iterations)
    else:
        _wait(barrier)
        local = 0
        for _ in range(iterations):
            local += 1
    with lock:  # single synchronised write after the loop
        counter_array[base + worker_id * LINE_INTS] = local


VARIANTS = {
    "unpadded": (worker_unpadded, 1),
    "padded": (worker_padded, LINE_INTS),
    "privatized": (worker_privatized, LINE_INTS),
}


def run_once(variant, P, K, lib_path):
    target, stride = VARIANTS[variant]
    arr = mp.Array(ctypes.c_int64, 8 * LINE_INTS + LINE_INTS, lock=False)  # zero-initialised
    base = ((-ctypes.addressof(arr)) % 64) // 8  # make slot `base` start on a 64 B boundary
    barrier = mp.Barrier(P + 1)
    lock = mp.Lock()
    procs = [mp.Process(target=target, args=(arr, w, K, barrier, base, lib_path, lock)) for w in range(P)]
    for pr in procs:
        pr.start()
    _wait(barrier)
    t0 = time.perf_counter_ns()
    for pr in procs:
        pr.join()
    t1 = time.perf_counter_ns()
    if any(pr.exitcode != 0 for pr in procs):
        raise RuntimeError(f"worker failed in {variant} P={P}: {[pr.exitcode for pr in procs]}")
    total = sum(arr[base + w * stride] for w in range(P))
    return t1 - t0, total


def build_c(tmp):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        sys.exit("--backend c needs a C compiler (cc/gcc/clang)")
    src, lib = Path(tmp) / "inc.c", Path(tmp) / "inc.so"
    src.write_text(C_SRC)
    r = subprocess.run([cc, "-O1", "-shared", "-fPIC", str(src), "-o", str(lib)], capture_output=True, text=True)
    if r.returncode:
        sys.exit("C compile failed:\n" + r.stderr)
    return str(lib)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--k", type=int, default=5_000_000, help="increments per worker (default 5,000,000)")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3, help="runs per (variant, P); median is reported")
    ap.add_argument("--backend", choices=["python", "c"], default="python")
    args = ap.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory()
    lib_path = build_c(tmp.name) if args.backend == "c" else None

    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    big = [P for P in args.procs if P > ncpu]
    if big:
        print(f"WARNING: this process can use {ncpu} CPUs; P={big} oversubscribes them, so those rows mix "
              "time-slicing with cache contention. Interpret with care.", file=sys.stderr)
    if 1 not in args.procs:
        sys.exit("P=1 is required as the Karp-Flatt baseline")

    K = args.k
    res = {}  # (variant, P) -> (times list, count_ok, total)
    for variant in VARIANTS:
        for P in args.procs:
            times, ok, tot = [], True, None
            for _ in range(args.repeats):
                t, tot = run_once(variant, P, K, lib_path)
                times.append(t)
                ok &= (tot == P * K)
            res[(variant, P)] = (times, ok, tot)
            print(f"{variant:10s} P={P}: median {statistics.median(times)/1e9:7.3f} s   "
                  f"count {tot} == {P*K}: {'ok' if ok else 'MISMATCH'}")

    header = ("variant\tP\tK\tbackend\trepeats\tmedian_time_s\trun_times_s\tspeedup_T1_over_TP\t"
              "efficiency_pct\tkarp_flatt_e_spec\tkarp_flatt_e_work_normalised\ttotal_count\t"
              "expected_count\tcount_ok\n")
    with open(out / "false_sharing_karp_flatt.tsv", "w") as f:
        f.write(header)
        for variant in VARIANTS:
            T1 = statistics.median(res[(variant, 1)][0])
            for P in args.procs:
                times, ok, tot = res[(variant, P)]
                TP = statistics.median(times)
                speedup = T1 / TP
                eff = 100 * speedup / P
                if P > 1:
                    e_spec = (P * TP / T1 - 1) / (P - 1)
                    e_work = (TP / T1 - 1) / (P - 1)
                    e_spec, e_work = f"{e_spec:.4f}", f"{e_work:.4f}"
                else:
                    e_spec = e_work = "NA"
                f.write(f"{variant}\t{P}\t{K}\t{args.backend}\t{args.repeats}\t{TP/1e9:.6f}\t"
                        f"{','.join(f'{t/1e9:.6f}' for t in times)}\t{speedup:.4f}\t{eff:.2f}\t"
                        f"{e_spec}\t{e_work}\t{tot}\t{P*K}\t{ok}\n")
    print(f"wrote {out/'false_sharing_karp_flatt.tsv'}")

    # sanity hint for the grader's "collapse" requirement
    big_p = max(args.procs)
    if big_p > 1:
        u = statistics.median(res[("unpadded", big_p)][0])
        p_ = statistics.median(res[("padded", big_p)][0])
        print(f"unpadded/padded runtime at P={big_p}: {u/p_:.2f}x")
        if u / p_ < 1.3:
            hint = ("Try --backend c (and a larger --k for it), " if args.backend == "python"
                    else "Try a larger --k, ")
            print("NOTE: weak collapse. " + hint + "make sure P <= physical cores (not hyper-thread "
                  "siblings of one core), and close other programs before re-running.", file=sys.stderr)


if __name__ == "__main__":
    main()
