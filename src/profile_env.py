#!/usr/bin/env python3
"""Milestone 0: host topology profiling + salted parameter derivation.

Usage:  python3 src/profile_env.py --student-id 12345678
Writes: results/host_profile.json, results/salted_params.json
"""
import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salt import get_student_params  # noqa: E402

try:
    import psutil
except ImportError:  # optional
    psutil = None


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def parse_size(text):
    """'32K' / '1M' / '48 KiB' / '8192' -> bytes (int) or None."""
    if text is None:
        return None
    m = re.match(r"\s*([\d.]+)\s*([KMG]?)i?B?", str(text).strip(), re.I)
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def read(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------- Linux
def linux_cpuinfo():
    info = {}
    text = read("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            info.setdefault(k, v)  # first processor entry
    return info, text


def linux_caches():
    """Read cpu0 cache hierarchy from sysfs. Returns dict level->(bytes), line size."""
    caches, line = {}, None
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        level = read(f"{d}/level")
        ctype = read(f"{d}/type")
        size = parse_size(read(f"{d}/size"))
        ls = read(f"{d}/coherency_line_size")
        if ls:
            line = int(ls)
        if level is None or size is None:
            continue
        if level == "1" and ctype == "Data":
            caches["l1d"] = size
        elif level == "1" and ctype == "Instruction":
            caches["l1i"] = size
        elif level == "2":
            caches["l2"] = size
        elif level == "3":
            caches["l3"] = size
    return caches, line


def lscpu_caches():
    """Fallback: parse `lscpu`. Reports per-instance size when '(N instances)' is present."""
    out, caches = sh(["lscpu"]), {}
    for key, name in (("L1d", "l1d"), ("L2", "l2"), ("L3", "l3")):
        m = re.search(rf"^{key} cache:\s+(.+)$", out, re.M)
        if not m:
            continue
        txt = m.group(1)
        size = parse_size(txt)
        inst = re.search(r"\((\d+) instances?\)", txt)
        if size and inst and name != "l3":  # lscpu prints totals on newer versions
            size //= int(inst.group(1))
        caches[name] = size
    return caches


def linux_physical_cores(cpuinfo_text):
    pairs = set()
    phys = core = None
    for line in cpuinfo_text.splitlines():
        if line.startswith("physical id"):
            phys = line.split(":")[1].strip()
        elif line.startswith("core id"):
            core = line.split(":")[1].strip()
            pairs.add((phys, core))
    return len(pairs) or None


# ----------------------------------------------------------------------------- macOS
def sysctl(name):
    out = sh(["sysctl", "-n", name]).strip()
    return out or None


# ----------------------------------------------------------------------------- main probe
def probe(microarch_override=None, base_ghz_override=None):
    system = platform.system()
    prof = {
        "os": system,
        "kernel": platform.release(),
        "kernel_version": platform.version(),
        "machine": platform.machine(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
    }
    try:
        import numpy
        prof["python"]["numpy"] = numpy.__version__
    except ImportError:
        pass

    model = vendor = None
    family = model_no = stepping = None
    freq_ghz, freq_src = None, None
    caches, line = {}, None
    physical = logical = None

    if system == "Linux":
        info, text = linux_cpuinfo()
        model = info.get("model name") or info.get("Model") or info.get("Hardware")
        vendor = info.get("vendor_id") or info.get("CPU implementer")
        family, model_no, stepping = info.get("cpu family"), info.get("model"), info.get("stepping")
        caches, line = linux_caches()
        if not caches.get("l1d") or not caches.get("l2"):
            for k, v in lscpu_caches().items():
                caches.setdefault(k, v)
        physical = linux_physical_cores(text)
        # base frequency: intel_pstate base_frequency (kHz) > '@ x.xxGHz' in model > lscpu max > cpuinfo (current!)
        bf = read("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
        if bf:
            freq_ghz, freq_src = int(bf) / 1e6, "sysfs base_frequency"
        elif model and (m := re.search(r"@\s*([\d.]+)\s*GHz", model)):
            freq_ghz, freq_src = float(m.group(1)), "model-name nominal"
        else:
            mx = re.search(r"CPU max MHz:\s*([\d.]+)", sh(["lscpu"]))
            cur = info.get("cpu MHz")
            if mx:
                freq_ghz, freq_src = float(mx.group(1)) / 1000, "lscpu max MHz (boost, not base)"
            elif cur:
                freq_ghz, freq_src = float(cur) / 1000, "/proc/cpuinfo current MHz (not base)"
    elif system == "Darwin":
        model = sysctl("machdep.cpu.brand_string")
        vendor = "Apple" if platform.machine() == "arm64" else sysctl("machdep.cpu.vendor")
        family, model_no, stepping = (sysctl("machdep.cpu.family"), sysctl("machdep.cpu.model"),
                                      sysctl("machdep.cpu.stepping"))
        for key, name in (("hw.l1dcachesize", "l1d"), ("hw.l2cachesize", "l2"), ("hw.l3cachesize", "l3")):
            v = sysctl(key)
            if v:
                caches[name] = int(v)
        v = sysctl("hw.cachelinesize")
        line = int(v) if v else None
        v = sysctl("hw.physicalcpu")
        physical = int(v) if v else None
        v = sysctl("hw.cpufrequency")  # Intel Macs only
        if v:
            freq_ghz, freq_src = int(v) / 1e9, "sysctl hw.cpufrequency"
    else:  # Windows / other
        model = platform.processor() or None
        if psutil:
            f = psutil.cpu_freq()
            if f:
                freq_ghz, freq_src = f.max / 1000 or f.current / 1000, "psutil"

    if psutil:
        physical = psutil.cpu_count(logical=False) or physical
        logical = psutil.cpu_count(logical=True)
    logical = logical or os.cpu_count()
    physical = physical or logical

    if base_ghz_override:
        freq_ghz, freq_src = base_ghz_override, "manual override"

    if microarch_override:
        uarch = microarch_override
    elif vendor == "Apple":
        uarch = "Apple Silicon (arm64)"
    else:
        uarch = f"unknown (vendor={vendor}, family={family}, model={model_no}, stepping={stepping}) " \
                "- pass --microarch to name it"

    prof.update({
        "cpu_model": model or platform.processor() or "unknown",
        "microarchitecture": uarch,
        "base_frequency_ghz": round(freq_ghz, 3) if freq_ghz else None,
        "base_frequency_source": freq_src,
        "physical_cores": physical,
        "logical_threads": logical,
        "smt_enabled": bool(logical and physical and logical > physical),
        "threads_per_core": (logical // physical) if physical else None,
        "usable_cpus_for_this_process": (len(os.sched_getaffinity(0))
                                         if hasattr(os, "sched_getaffinity") else logical),
        "cache": {
            "l1d_bytes": caches.get("l1d"),
            "l2_bytes": caches.get("l2"),
            "l3_bytes": caches.get("l3"),
            "line_size_bytes": line or 64,
            "line_size_source": "sysfs/sysctl" if line else "assumed 64",
            "note": "L1d/L2 are per-core, L3 is shared across cores (per-instance values).",
        },
        "total_ram_bytes": psutil.virtual_memory().total if psutil else None,
    })
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-id", required=True, help="8-digit student ID")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--microarch", help="override microarchitecture name, e.g. 'Zen 3', 'Alder Lake P-core'")
    ap.add_argument("--base-ghz", type=float, help="override base frequency in GHz")
    args = ap.parse_args()

    if not re.fullmatch(r"\d{8}", args.student_id):
        print(f"WARNING: student id '{args.student_id}' is not 8 digits; check it before submitting.",
              file=sys.stderr)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    prof = probe(args.microarch, args.base_ghz)
    (out / "host_profile.json").write_text(json.dumps(prof, indent=2) + "\n")
    params = get_student_params(args.student_id)
    (out / "salted_params.json").write_text(json.dumps(params, indent=2) + "\n")

    c = prof["cache"]
    print(f"CPU: {prof['cpu_model']}  [{prof['microarchitecture']}]")
    print(f"cores: {prof['physical_cores']} physical / {prof['logical_threads']} logical "
          f"(SMT={'on' if prof['smt_enabled'] else 'off'}), base {prof['base_frequency_ghz']} GHz "
          f"({prof['base_frequency_source']})")
    print(f"cache: L1d={c['l1d_bytes']} L2={c['l2_bytes']} L3={c['l3_bytes']} line={c['line_size_bytes']} B")
    print(f"salted params: {params}")
    missing = [k for k in ("l1d_bytes", "l2_bytes", "l3_bytes") if not c[k]]
    if missing or not prof["base_frequency_ghz"] or prof["microarchitecture"].startswith("unknown"):
        print("NOTE: some fields could not be detected automatically "
              f"(missing caches: {missing}); edit results/host_profile.json by hand or use "
              "--microarch/--base-ghz before moving on.", file=sys.stderr)
    print(f"wrote {out/'host_profile.json'} and {out/'salted_params.json'}")


if __name__ == "__main__":
    main()
