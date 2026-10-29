#!/usr/bin/env python3
"""Milestone 1, Task 1.2: loop interchange (i-j-k vs i-k-j) on a dense matmul kernel.

A, B are N x N (salted N) float64 matrices stored as FLAT row-major buffers, so
B[k, j] is B[k*N + j] and walking k with j fixed strides by N elements.  Both
algorithms run on the top-left M x M sub-block, M = N // 2.

Backend:
  * default "c": the two loop nests are compiled once with the system C compiler
    (-O1, no vectorisation / FMA contraction / auto-interchange, so the compiler
    does not undo the effect we are measuring) and called through ctypes.
  * fallback "python": pure-Python loops over flat lists (very slow; use --m to shrink).

Both variants accumulate over k in the same order, so C_naive and C_interchanged are
bit-for-bit identical (reported as max_abs_diff = 0.0).
Output: results/loop_interchange.json
"""
import argparse
import ctypes
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salt import load_params  # noqa: E402

C_SRC = r"""
void matmul_ijk(const double *A, const double *B, double *C, long N, long M) {
    for (long i = 0; i < M; i++)
        for (long j = 0; j < M; j++) {
            double s = 0.0;
            for (long k = 0; k < M; k++)
                s += A[i*N + k] * B[k*N + j];      /* B walks down a column: stride N */
            C[i*N + j] = s;
        }
}
void matmul_ikj(const double *A, const double *B, double *C, long N, long M) {
    for (long i = 0; i < M; i++) {
        for (long j = 0; j < M; j++) C[i*N + j] = 0.0;
        for (long k = 0; k < M; k++) {
            double a = A[i*N + k];                 /* hoisted into a register */
            for (long j = 0; j < M; j++)
                C[i*N + j] += a * B[k*N + j];      /* stride-1 over B row and C row */
        }
    }
}
"""
CFLAGS = ["-O1", "-ffp-contract=off", "-fno-tree-vectorize", "-shared", "-fPIC"]


def build_c_backend(tmp):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    src = Path(tmp) / "kernel.c"
    lib = Path(tmp) / "kernel.so"
    src.write_text(C_SRC)
    r = subprocess.run([cc, *CFLAGS, str(src), "-o", str(lib)], capture_output=True, text=True)
    if r.returncode != 0:
        print("C compile failed, falling back to pure Python:\n" + r.stderr, file=sys.stderr)
        return None
    dll = ctypes.CDLL(str(lib))
    ptr = ctypes.POINTER(ctypes.c_double)
    for name in ("matmul_ijk", "matmul_ikj"):
        fn = getattr(dll, name)
        fn.argtypes = [ptr, ptr, ptr, ctypes.c_long, ctypes.c_long]
        fn.restype = None
    return dll, cc


def py_ijk(A, B, C, N, M):
    for i in range(M):
        for j in range(M):
            s = 0.0
            for k in range(M):
                s += A[i * N + k] * B[k * N + j]
            C[i * N + j] = s


def py_ikj(A, B, C, N, M):
    for i in range(M):
        rowC = i * N
        for j in range(M):
            C[rowC + j] = 0.0
        for k in range(M):
            a = A[i * N + k]
            rowB = k * N
            for j in range(M):
                C[rowC + j] += a * B[rowB + j]


def timed(fn, runs):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    return ts


def digest(Csub):
    return hashlib.sha256(np.round(Csub, 6).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-id", help="recompute salted params instead of reading results/salted_params.json")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--backend", choices=["auto", "c", "python"], default="auto")
    ap.add_argument("--m", type=int, help="override sub-block size M (default N//2)")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = load_params(args.results_dir, args.student_id)
    N, seed = p["N"], p["seed"]
    M = args.m or N // 2
    assert 0 < M <= N

    rng = np.random.default_rng(seed)
    A = rng.random(N * N)  # flat row-major buffers
    B = rng.random(N * N)
    C1 = np.zeros(N * N)
    C2 = np.zeros(N * N)

    backend, compiler = args.backend, None
    tmp = tempfile.TemporaryDirectory()
    dll = None
    if backend in ("auto", "c"):
        built = build_c_backend(tmp.name)
        if built:
            dll, compiler = built
            backend = "c"
        elif backend == "c":
            sys.exit("no working C compiler found; use --backend python")
        else:
            backend = "python"

    print(f"N={N} M={M} seed={seed} backend={backend} runs={args.runs}")
    if backend == "c":
        dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        f_naive = lambda: dll.matmul_ijk(dp(A), dp(B), dp(C1), N, M)
        f_inter = lambda: dll.matmul_ikj(dp(A), dp(B), dp(C2), N, M)
        f_naive(); f_inter()  # untimed warm-up (page faults, caches)
    else:
        print(f"WARNING: pure Python does {M**3:,} multiply-adds per run per algorithm; this will be slow.",
              file=sys.stderr)
        Al, Bl = A.tolist(), B.tolist()
        C1l, C2l = [0.0] * (N * N), [0.0] * (N * N)
        f_naive = lambda: py_ijk(Al, Bl, C1l, N, M)
        f_inter = lambda: py_ikj(Al, Bl, C2l, N, M)

    t_naive = timed(f_naive, args.runs)
    t_inter = timed(f_inter, args.runs)
    if backend == "python":
        C1, C2 = np.array(C1l), np.array(C2l)

    S1 = C1.reshape(N, N)[:M, :M]
    S2 = C2.reshape(N, N)[:M, :M]
    ref = A.reshape(N, N)[:M, :M] @ B.reshape(N, N)[:M, :M]  # informational only
    max_diff = float(np.max(np.abs(S1 - S2)))
    max_diff_numpy = float(np.max(np.abs(S1 - ref)))

    med_n, med_i = statistics.median(t_naive), statistics.median(t_inter)
    speedup = med_n / med_i
    res = {
        "student_id": p["student_id"], "N": N, "M": M, "seed": seed,
        "backend": backend, "compiler": compiler, "cflags": CFLAGS if backend == "c" else None,
        "runs": args.runs,
        "naive_ijk_times_ns": t_naive, "interchanged_ikj_times_ns": t_inter,
        "naive_ijk_median_ns": med_n, "interchanged_ikj_median_ns": med_i,
        "naive_ijk_median_s": med_n / 1e9, "interchanged_ikj_median_s": med_i / 1e9,
        "speedup_naive_over_interchanged": speedup,
        "max_abs_diff_naive_vs_interchanged": max_diff,
        "tolerance": 1e-12, "within_tolerance": max_diff <= 1e-12,
        "max_abs_diff_vs_numpy_matmul_informational": max_diff_numpy,
        "sha256_round6_naive": digest(S1), "sha256_round6_interchanged": digest(S2),
        "checksum_sum_C": float(S1.sum()),
    }
    (out / "loop_interchange.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"naive i-j-k     median {med_n/1e9:8.3f} s")
    print(f"interchanged    median {med_i/1e9:8.3f} s")
    print(f"speedup S = {speedup:.2f}x   max|C_naive - C_interchanged| = {max_diff:.3g}")
    print(f"wrote {out/'loop_interchange.json'}")
    if not res["within_tolerance"]:
        sys.exit("ERROR: interchanged result differs from baseline beyond 1e-12")


if __name__ == "__main__":
    main()
