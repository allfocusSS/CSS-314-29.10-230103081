# Karp–Flatt Practice Lab: Results

## Hardware profile

- Logical cores (`os.cpu_count()`): **2**
- Python 3.13, Linux (cloud sandbox VM)
- P=4 therefore **oversubscribes** the machine (4 processes on 2 cores). The handout assumes at least 4 cores.

## Method

Each script was run 4 times. Values below are the **median** across runs (T and e computed per run, then medianed).
Raw per-run output for the first run: A (4.653 / 2.829 / 2.662 s), B (1.360 / 10.018 / 4.353 s), C (0.710 / 0.344 / 0.417 s).

## Results

| Experiment | Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Diagnosis |
|---|---|---|---|---|---|---|---|
| A: Serial Tail | 0.40 s sleep + 16 chunks | 4.654 | 2.793 | 2.656 | 0.211 | 0.428 | Serial fraction visible at P=2; P=4 distorted by 2-core limit |
| B: Lock Contention | shared Value + Lock | 1.299 | 9.925 | 4.875 | 13.983 | 4.466 | Severe lock thrashing: T2 and T4 are both far above T1 |
| C: Privatized State | private accumulators, one reduction | 0.713 | 0.433 | 0.391 | 0.115 | 0.425 | Near-linear scaling at P=2; P=4 limited by 2 cores |

## Rubric self-check

| Check | Measured | Verdict |
|---|---|---|
| A: e(4)/e(2) in [0.80, 1.25] | 2.05 (runs ranged 1.99 to 3.44) | **Not met** on this machine |
| B: T4 > T1 and e(4) >> e(2) | T4 > T1 holds (4.88 s vs 1.30 s); e(4) < e(2) | **Partly met**: slowdown is clear, but e(4) does not exceed e(2) |
| C: T4 < T1 and E4 > 70% | T4 < T1 holds; E4 = 44% | **Partly met**: speedup exists, efficiency target not met |

## Why the numbers differ from the handout

1. **Only 2 cores.** With P=4 on 2 cores, the extra two processes cannot run in parallel. T4 is bounded by roughly T2, so e(4) is inflated by oversubscription and not by the algorithm. This is why A's e(4) is about double e(2) and why C's E4 is capped near 50%. On a machine with at least 4 physical cores, the A ratio should land near 1 and C's efficiency should be much higher.
2. **Experiment A's e(2) is above the handout's 0.10–0.14.** The sleep is 0.40 s out of about 4.65 s total, so the sequential fraction is about 0.086 of T1. Pool startup and the final `map` gather add further serial cost, which pushes e(2) to roughly 0.12–0.24 across runs.
3. **Experiment B, e(4) below e(2).** With 2 cores, two processes contend on the lock continuously, each preempting the other while holding it. With 4 processes the 2 cores stay busy with whichever process holds the lock, so total time drops from T2 to T4. This pattern is specific to the 2-core case. It still shows the contention pathology (T2 is about 7.6x T1), but it does not reproduce the monotonic growth of e(p) from the handout.
4. **Experiment C timing noise.** T1 is only about 0.7 s, and one run measured 1.55 s at P=1, which produced a negative e(2) in that run. The median is more reliable than any single run.

## Conclusion

- A shows a fixed serial component, but the P=4 point cannot be read cleanly on 2 cores.
- B shows the clearest result: with a shared lock, adding processes slows the program down by roughly 4x to 8x compared with one process.
- C confirms that privatizing state restores scaling up to the physical core count (P=2 gives 1.6x speedup at median).

To meet the rubric as written, rerun all three scripts on a machine reporting `os.cpu_count() >= 4` and replace the table above with those timings.
