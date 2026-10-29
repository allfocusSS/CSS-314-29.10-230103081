#!/usr/bin/env python3
"""Milestone 3: strong / weak scaling of a 5-point Jacobi stencil + Roofline placement.

Kernel (one sweep over the interior, borders fixed):
    new[i,j] = 0.25 * (old[i-1,j] + old[i+1,j] + old[i,j-1] + old[i,j+1])      -> 4 FLOPs / point
Interior rows are split into P contiguous chunks, one worker process per chunk; the two grids
live in multiprocessing.shared_memory, and a barrier separates sweeps (Jacobi needs the whole
previous sweep).  The timed region (barrier release -> final checksum) also contains the serial
checksum reduction done by the parent, so the measured serial fraction is "honest".

Strong scaling : fixed N (salted), P in {1,2,4,...,max physical cores}.
Weak scaling   : N_P = ceil(N * sqrt(P)), so work grows as O(P).
Fits           : Amdahl  s = (1/S_P - 1/P) / (1 - 1/P)   (least-squares over all P>1)
                 Gustafson  S_scaled = P - g (P - 1)
Roofline       : arithmetic intensity, machine ceilings, regime classification.

Outputs: results/scaling_metrics.tsv, results/scaling_fit.json, results/roofline_summary.json
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salt import load_params  # noqa: E402

FLOPS_PER_POINT = 4  # 3 adds + 1 multiply


def sweep(src, dst, r0, r1):
    """Update rows r0..r1-1 (all interior columns) of dst from src. Fixed op order => bitwise reproducible."""
    tmp = np.add(src[r0 - 1:r1 - 1, 1:-1], src[r0 + 1:r1 + 1, 1:-1])
    np.add(tmp, src[r0:r1, 0:-2], out=tmp)
    np.add(tmp, src[r0:r1, 2:], out=tmp)
    np.multiply(tmp, 0.25, out=dst[r0:r1, 1:-1])


def worker(name_a, name_b, N, r0, r1, iters, start_barrier, iter_barrier):
    sa, sb = shared_memory.SharedMemory(name=name_a), shared_memory.SharedMemory(name=name_b)
    bufs = (np.ndarray((N, N), dtype=np.float64, buffer=sa.buf),
            np.ndarray((N, N), dtype=np.float64, buffer=sb.buf))
    start_barrier.wait(timeout=900)
    for t in range(iters):
        sweep(bufs[t & 1], bufs[(t + 1) & 1], r0, r1)
        iter_barrier.wait(timeout=900)
    del bufs
    sa.close()
    sb.close()


def chunks(N, P):
    rows = N - 2  # interior rows 1..N-2
    edges = [1 + (rows * p) // P for p in range(P + 1)]
    return list(zip(edges[:-1], edges[1:]))


def init_grid(N, seed):
    return np.random.default_rng(seed).random((N, N))


def reference(N, iters, seed):
    """Single-threaded reference result (same arithmetic, whole interior in one call)."""
    a, b = init_grid(N, seed), init_grid(N, seed)
    bufs = (a, b)
    for t in range(iters):
        sweep(bufs[t & 1], bufs[(t + 1) & 1], 1, N - 1)
    final = bufs[iters & 1]
    return float(np.sum(final)), final


def digest(final):
    return hashlib.sha256(np.round(final, 6).tobytes()).hexdigest()


def run_parallel(N, P, iters, seed):
    nbytes = N * N * 8
    sa, sb = shared_memory.SharedMemory(create=True, size=nbytes), shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        a = np.ndarray((N, N), dtype=np.float64, buffer=sa.buf)
        b = np.ndarray((N, N), dtype=np.float64, buffer=sb.buf)
        a[:] = init_grid(N, seed)
        b[:] = a  # identical borders in both buffers
        start_b, iter_b = mp.Barrier(P + 1), mp.Barrier(P)
        procs = [mp.Process(target=worker, args=(sa.name, sb.name, N, r0, r1, iters, start_b, iter_b))
                 for r0, r1 in chunks(N, P)]
        for pr in procs:
            pr.start()
        start_b.wait(timeout=900)
        t0 = time.perf_counter_ns()
        for pr in procs:
            pr.join()
        final = a if (iters & 1) == 0 else b
        checksum = float(np.sum(final))  # serial reduction (part of the timed region)
        t1 = time.perf_counter_ns()
        if any(pr.exitcode != 0 for pr in procs):
            raise RuntimeError(f"worker failure: {[pr.exitcode for pr in procs]}")
        dg = digest(final)
        del a, b, final
        return t1 - t0, checksum, dg
    finally:
        for s in (sa, sb):
            s.close()
            s.unlink()


def measure_copy_bandwidth_gbs(mb=256, reps=5):
    """Single-thread STREAM-copy-like bandwidth (read+write bytes only; lower bound on peak)."""
    n = mb * 1024 * 1024 // 8
    x, y = np.ones(n), np.empty(n)
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        np.copyto(y, x)
        dt = time.perf_counter() - t0
        best = max(best, 2 * n * 8 / dt / 1e9)
    return best


def lsq_through_origin(x, y):
    sxx = sum(v * v for v in x)
    return sum(a * b for a, b in zip(x, y)) / sxx if sxx else float("nan")


def p_list(max_p):
    ps, p = [], 1
    while p <= max_p:
        ps.append(p)
        p *= 2
    if ps[-1] != max_p:
        ps.append(max_p)
    return ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-id")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--iters", type=int, default=50, help="Jacobi sweeps per run")
    ap.add_argument("--repeats", type=int, default=3, help="runs per configuration; median is used")
    ap.add_argument("--max-p", type=int, help="override max worker count (default: physical cores)")
    ap.add_argument("--n", type=int, help="override salted N (testing only)")
    ap.add_argument("--peak-bw-gbs", type=float,
                    help="theoretical peak DRAM bandwidth of your machine in GB/s (e.g. DDR4-3200 dual channel = 51.2)")
    ap.add_argument("--peak-gflops", type=float, help="override peak DP GFLOP/s (default: cores*GHz*flops_per_cycle)")
    ap.add_argument("--flops-per-cycle", type=float, default=16.0,
                    help="DP FLOPs/cycle/core for the peak estimate (16 = AVX2, 2 FMA units; 32 for AVX-512)")
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    p = load_params(args.results_dir, args.student_id)
    N, seed = args.n or p["N"], p["seed"]

    prof_path = res_dir / "host_profile.json"
    prof = json.loads(prof_path.read_text()) if prof_path.exists() else {}
    phys = prof.get("physical_cores") or os.cpu_count()
    max_p = args.max_p or phys
    Ps = p_list(max_p)
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if max_p > ncpu:
        print(f"WARNING: max P={max_p} exceeds the {ncpu} CPUs available to this process.", file=sys.stderr)
    print(f"N={N} seed={seed} iters={args.iters} P list={Ps}")

    ref_cache = {}

    def ref_for(n):
        if n not in ref_cache:
            cs, final = reference(n, args.iters, seed)
            ref_cache[n] = (cs, digest(final))
        return ref_cache[n]

    def median_run(n, P):
        ts, cs, dg = [], None, None
        for _ in range(args.repeats):
            t, cs, dg = run_parallel(n, P, args.iters, seed)
            ts.append(t)
        return ts, cs, dg

    rows = []  # dict per row
    # ---- strong scaling
    T1 = None
    for P in Ps:
        ts, cs, dg = median_run(N, P)
        T = statistics.median(ts)
        T1 = T1 or T
        ref_cs, ref_dg = ref_for(N)
        S = T1 / T
        rows.append(dict(mode="strong", P=P, N=N, work=N * N, T=T, times=ts, speedup=S, eff=100 * S / P,
                         scaled="NA", scaled_lit="NA",
                         s_est=((1 / S - 1 / P) / (1 - 1 / P)) if P > 1 else "NA", g_est="NA",
                         cs=cs, ref=ref_cs, ok=(cs == ref_cs and dg == ref_dg)))
        print(f"strong P={P:2d} N={N}: {T/1e9:8.3f} s  S={S:5.2f}  E={100*S/P:5.1f}%  checksum "
              f"{'ok' if rows[-1]['ok'] else 'MISMATCH'}")

    # ---- weak scaling
    W1 = N * N
    T1w = None
    for P in Ps:
        NP = math.ceil(N * math.sqrt(P))
        ts, cs, dg = median_run(NP, P)
        T = statistics.median(ts)
        T1w = T1w or T
        ref_cs, ref_dg = ref_for(NP)
        work = NP * NP
        scaled = (work / W1) * (T1w / T)                      # standard scaled speedup
        scaled_lit = (P * W1 * T1w) / (work * T)               # literal expression from the handout
        rows.append(dict(mode="weak", P=P, N=NP, work=work, T=T, times=ts, speedup="NA",
                         eff=100 * T1w / T, scaled=scaled, scaled_lit=scaled_lit, s_est="NA",
                         g_est=((P - scaled) / (P - 1)) if P > 1 else "NA",
                         cs=cs, ref=ref_cs, ok=(cs == ref_cs and dg == ref_dg)))
        print(f"weak   P={P:2d} N={NP}: {T/1e9:8.3f} s  scaled S={scaled:5.2f}  weak eff={100*T1w/T:5.1f}%  "
              f"checksum {'ok' if rows[-1]['ok'] else 'MISMATCH'}")

    def fmt(v, d=4):
        return v if isinstance(v, str) else f"{v:.{d}f}"

    with open(res_dir / "scaling_metrics.tsv", "w") as f:
        f.write("mode\tP\tN\twork_points\titers\tmedian_time_s\trun_times_s\tspeedup_T1_over_TP\t"
                "efficiency_pct\tscaled_speedup\tscaled_speedup_handout_literal\tamdahl_s_estimate\t"
                "gustafson_s_estimate\tchecksum\treference_checksum\tchecksum_match\n")
        for r in rows:
            f.write(f"{r['mode']}\t{r['P']}\t{r['N']}\t{r['work']}\t{args.iters}\t{r['T']/1e9:.6f}\t"
                    f"{','.join(f'{t/1e9:.6f}' for t in r['times'])}\t{fmt(r['speedup'])}\t{fmt(r['eff'],2)}\t"
                    f"{fmt(r['scaled'])}\t{fmt(r['scaled_lit'])}\t{fmt(r['s_est'])}\t{fmt(r['g_est'])}\t"
                    f"{r['cs']!r}\t{r['ref']!r}\t{r['ok']}\n")

    # ---- model fits
    strong = [r for r in rows if r["mode"] == "strong" and r["P"] > 1]
    weak = [r for r in rows if r["mode"] == "weak" and r["P"] > 1]
    fit = {"N": N, "iters": args.iters, "P_list": Ps}
    if strong:
        x = [1 - 1 / r["P"] for r in strong]
        y = [1 / r["speedup"] - 1 / r["P"] for r in strong]
        s = lsq_through_origin(x, y)
        fit["amdahl"] = {
            "serial_fraction_s_lsq": s,
            "serial_fraction_s_at_each_P": {r["P"]: r["s_est"] for r in strong},
            "max_asymptotic_speedup_1_over_s": (1 / s) if s > 0 else None,
            "note": "s <= 0 means super-linear or noise-dominated measurements" if s <= 0 else
                    "s lumps true serial work (init/reduction) with barrier cost and memory-bandwidth saturation",
        }
    if weak:
        x = [r["P"] - 1 for r in weak]
        y = [r["P"] - r["scaled"] for r in weak]
        fit["gustafson"] = {
            "serial_fraction_g_lsq": lsq_through_origin(x, y),
            "scaled_speedup_at_each_P": {r["P"]: r["scaled"] for r in weak},
            "weak_efficiency_pct_at_each_P": {r["P"]: r["eff"] for r in weak},
        }
    if strong and weak:
        pm = max(r["P"] for r in strong)
        es = next(r for r in strong if r["P"] == pm)["eff"]
        ew = next(r for r in weak if r["P"] == pm)["eff"]
        fit["contrast"] = {
            "P": pm, "strong_efficiency_pct": es, "weak_efficiency_pct": ew,
            "weak_minus_strong_efficiency_pct_points": ew - es,
            "note": "Interpret yourself: weak scaling normally keeps efficiency higher than strong scaling because "
                    "per-process work stays constant, but a bandwidth-bound kernel or a cache-size change between "
                    "N and N_P can reverse that.",
        }
    (res_dir / "scaling_fit.json").write_text(json.dumps(fit, indent=2) + "\n")

    # ---- roofline
    cache = prof.get("cache", {})
    l3 = cache.get("l3_bytes")
    work_bytes = 2 * N * N * 8
    bytes_wa = 24  # read src 8 + write dst 8 + write-allocate read of dst line 8
    bytes_nt = 16  # with non-temporal stores
    ai = FLOPS_PER_POINT / bytes_wa
    ai_nt = FLOPS_PER_POINT / bytes_nt
    ghz = prof.get("base_frequency_ghz")
    if args.peak_gflops:
        peak_gf, gf_src = args.peak_gflops, "user supplied"
    elif ghz and phys:
        peak_gf = phys * ghz * args.flops_per_cycle
        gf_src = f"estimate: {phys} cores x {ghz} GHz x {args.flops_per_cycle:g} DP FLOP/cycle (assumption!)"
    else:
        peak_gf, gf_src = None, "unknown (pass --peak-gflops)"
    meas_bw = measure_copy_bandwidth_gbs()
    if args.peak_bw_gbs:
        peak_bw, bw_src = args.peak_bw_gbs, "user supplied theoretical peak"
    else:
        peak_bw, bw_src = meas_bw, "measured single-thread copy bandwidth (LOWER BOUND of true peak)"
    ridge = (peak_gf / peak_bw) if peak_gf else None
    big = max(strong, key=lambda r: r["P"]) if strong else rows[0]
    pts_per_s = big["work"] * args.iters / (big["T"] / 1e9)
    achieved_gf = pts_per_s * FLOPS_PER_POINT / 1e9
    achieved_gbs = pts_per_s * bytes_wa / 1e9
    bound = ("memory-bandwidth bound" if (ridge is None or ai < ridge) else "compute bound")
    summary = {
        "kernel": "5-point Jacobi stencil, float64",
        "flops_per_point": FLOPS_PER_POINT,
        "dram_bytes_per_point": {"streaming_with_write_allocate": bytes_wa, "with_nontemporal_stores": bytes_nt,
                                 "assumption": "each source value is loaded from DRAM once (neighbouring rows "
                                               "are reused from cache), each destination value written once"},
        "arithmetic_intensity_flop_per_byte": ai,
        "arithmetic_intensity_nontemporal_stores": ai_nt,
        "machine": {
            "peak_gflops": peak_gf, "peak_gflops_source": gf_src,
            "peak_bandwidth_gbs": peak_bw, "peak_bandwidth_source": bw_src,
            "measured_single_thread_copy_bandwidth_gbs": meas_bw,
            "ridge_point_flop_per_byte": ridge,
            "cache_bytes": {"l1d": cache.get("l1d_bytes"), "l2": cache.get("l2_bytes"), "l3": l3},
        },
        "working_set_bytes_two_grids": work_bytes,
        "working_set_fits_in_l3": (work_bytes <= l3) if l3 else None,
        "achieved": {"P": big["P"], "gflops": achieved_gf, "effective_bandwidth_gbs_model": achieved_gbs,
                     "fraction_of_bandwidth_roof": (achieved_gbs / peak_bw) if peak_bw else None},
        "regime": bound,
        "conclusion": (
            f"AI = {ai:.3f} FLOP/B vs ridge point "
            + (f"{ridge:.2f} FLOP/B" if ridge else "unknown")
            + f" -> {bound}. Bandwidth roof at this AI = {ai*peak_bw:.2f} GFLOP/s"
            + (f", compute peak = {peak_gf:.0f} GFLOP/s." if peak_gf else ".")
            + (" The two grids fit in L3, so the relevant bandwidth is L3 rather than DRAM."
               if (l3 and work_bytes <= l3) else
               " The two grids exceed L3, so each sweep streams from DRAM." if l3 else "")
            + " Check peak_bandwidth_source: a single-thread copy measurement understates the multi-core peak."
        ),
    }
    (res_dir / "roofline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"AI={ai:.3f} FLOP/B  ridge={ridge}  regime={bound}")
    print(f"wrote scaling_metrics.tsv, scaling_fit.json, roofline_summary.json in {res_dir}")
    if not all(r["ok"] for r in rows):
        sys.exit("ERROR: parallel checksum differs from the single-threaded reference")


if __name__ == "__main__":
    main()
