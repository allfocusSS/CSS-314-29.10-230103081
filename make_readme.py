#!/usr/bin/env python3
"""Generate README.md (one-paragraph bottleneck summary) from the files in results/.

Every number in the paragraph is read from your own result files, so re-run this after any
re-measurement.  Usage:  python3 make_readme.py [--results-dir results]
"""
import argparse
import csv
import json
from pathlib import Path


def jload(p):
    return json.loads(p.read_text()) if p.exists() else {}


def tsv(p):
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--no-ai-note", action="store_true", help="omit the AI-assistance disclosure line")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "README.md"), help="output README path")
    args = ap.parse_args()
    R = Path(args.results_dir)

    host, params = jload(R / "host_profile.json"), jload(R / "salted_params.json")
    loop, fit, roof = jload(R / "loop_interchange.json"), jload(R / "scaling_fit.json"), jload(R / "roofline_summary.json")
    stride, fs = tsv(R / "stride_results.tsv"), tsv(R / "false_sharing_karp_flatt.tsv")
    cache = host.get("cache", {})
    line = cache.get("line_size_bytes") or 64
    parts = []

    # ---- M1 stride
    if stride:
        by = {int(r["stride"]): num(r["latency_ns_per_element"]) for r in stride}
        ls = line // 8
        cand = [s for s in sorted(by) if s >= ls]
        s_line = cand[0] if cand else max(by)
        parts.append(
            f"On {host.get('cpu_model', 'this CPU')} the cost per element read rises from {by[1]:.2f} ns at stride 1 to "
            f"{by[s_line]:.2f} ns at stride {s_line} (one element per {line}-byte cache line), a {by[s_line]/by[1]:.1f}x "
            f"increase, and is {by[max(by)]:.2f} ns at stride {max(by)}"
            + (", so a strided traversal wastes most of every fetched line and is limited by memory rather than arithmetic."
               if by[s_line] / by[1] >= 1.5 else "; the cliff is weak on this machine, which is a measurement caveat."))
    if loop:
        parts.append(
            f"Interchanging the dense kernel from i-j-k to i-k-j (N={loop['N']}, M={loop['M']}, {loop['backend']} backend) turns "
            f"the stride-N walk over B into stride-1 accesses and gives a {loop['speedup_naive_over_interchanged']:.2f}x speedup "
            f"with bit-identical results.")

    # ---- M2 false sharing
    if fs:
        def row(var, P):
            for r in fs:
                if r["variant"] == var and int(r["P"]) == P:
                    return r
        Ps = sorted({int(r["P"]) for r in fs})
        multi = [P for P in Ps if P > 1]
        if multi:
            P = 4 if 4 in multi else multi[-1]
            u, p_, pr = row("unpadded", P), row("padded", P), row("privatized", P)
            tu, tp, tpr = (num(x["median_time_s"]) for x in (u, p_, pr))
            u1, p1 = num(row("unpadded", 1)["median_time_s"]), num(row("padded", 1)["median_time_s"])
            backend = u.get("backend", "python")
            txt = (f"With {line}-byte lines, counters packed on one line take {tu:.3f} s at P={P} versus {tp:.3f} s when padded "
                   f"to one line each ({tu/tp:.1f}x slower; T(P)/T(1) = {tu/u1:.1f} unpadded vs {tp/p1:.1f} padded for "
                   f"equal work per worker), while privatization takes {tpr:.3f} s. Karp-Flatt e({P}) is "
                   f"{num(u['karp_flatt_e_spec']):.2f} unpadded vs {num(p_['karp_flatt_e_spec']):.2f} padded by the handout "
                   f"formula ({num(u['karp_flatt_e_work_normalised']):.2f} vs {num(p_['karp_flatt_e_work_normalised']):.2f} "
                   f"with work-normalised speedup)"
                   + (", diagnosing cache-line ping-pong as the unpadded bottleneck." if tu / tp >= 1.3 else
                      "; at this P the unpadded penalty is small, so contention is not clearly established."))
            if backend == "c":
                txt += (" The increment loop of this run executes in compiled C via ctypes because CPython's per-iteration "
                        "overhead masked most of the contention in a pure-Python run.")
            if "Apple" in str(host.get("microarchitecture", "")):
                txt += (" Rows with P above the performance-core count may include slower efficiency cores, so "
                        "the cleanest comparison is at small P.")
            parts.append(txt)

    # ---- M3 scaling + roofline
    am, gu, ct = fit.get("amdahl", {}), fit.get("gustafson", {}), fit.get("contrast", {})
    if am and ct:
        s = am["serial_fraction_s_lsq"]
        smax = am.get("max_asymptotic_speedup_1_over_s")
        parts.append(
            f"The stencil's strong scaling is {'poor' if s > 0.3 else 'reasonable'}: the Amdahl fit gives an effective serial fraction s = {s:.2f}"
            + (f" (S_max = {smax:.1f}x)" if smax else "")
            + f", with {ct['strong_efficiency_pct']:.0f}% efficiency at P={ct['P']}, versus "
              f"{ct['weak_efficiency_pct']:.0f}% weak-scaling efficiency at P={ct['P']} (Gustafson g = {gu.get('serial_fraction_g_lsq', float('nan')):.2f}); "
              f"the effective s lumps true serial work with barrier cost and memory-bandwidth saturation.")
    if roof:
        m = roof["machine"]
        ridge = m.get("ridge_point_flop_per_byte")
        parts.append(
            f"Its arithmetic intensity is {roof['arithmetic_intensity_flop_per_byte']:.3f} FLOP/byte"
            + (f", well below the ridge point of {ridge:.1f} FLOP/byte," if ridge and roof['arithmetic_intensity_flop_per_byte'] < ridge else ",")
            + f" so the kernel is {roof['regime']} (peak GFLOP/s {m.get('peak_gflops')} and bandwidth "
              f"{m.get('peak_bandwidth_gbs'):.0f} GB/s are estimates: {m.get('peak_gflops_source')}; {m.get('peak_bandwidth_source')})."
            + ("" if cache.get("l3_bytes") else " This CPU reports no L3 cache, so no L3 comparison is made."))

    sid = params.get("student_id", "<YOUR_ID>")
    body = " ".join(parts)
    text = f"# CS482 Performance Practicum - Student {sid}\n\n{body}\n"
    if not args.no_ai_note:
        text += ("\n_Disclosure: the measurement scripts were written with AI assistance (Claude), which the "
                 "instructor permitted; all numbers above come from my own runs._\n")
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
