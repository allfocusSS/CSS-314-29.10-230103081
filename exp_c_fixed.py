import time
from multiprocessing import Pool


def private_worker(iterations):
    # Accumulate in private memory (zero shared state)
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # Synchronize exactly ONCE at the end
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {p: run_privatized(p) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        eff = sp / p
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | KarpFlatt e={e:.3f} | Efficiency={eff:.0%}")
