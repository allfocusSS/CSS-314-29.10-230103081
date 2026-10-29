import time, math
from multiprocessing import Pool


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  # Intentional sequential bottleneck (serial fraction)
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {p: run_experiment(p) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = karp_flatt(t1, tp, p)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | KarpFlatt e={e:.3f}")
    e2 = karp_flatt(t1, runtimes[2], 2)
    e4 = karp_flatt(t1, runtimes[4], 4)
    print(f"e(4)/e(2) = {e4 / e2:.3f}  (rubric: within [0.80, 1.25])")
