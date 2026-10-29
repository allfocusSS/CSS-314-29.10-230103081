# Karp–Flatt Practice Lab: Results

## Hardware profile

- Machine: MacBook Air, chip: **Apple M2**
- Logical cores (`os.cpu_count()`): **8**
- Python: **3.13.0**, macOS

## Method

Each script was run 3 times. Values in the table are the **median** across runs (T taken directly, e computed per run and then medianed).

## Results

| Experiment | Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|---|
| A: Serial Tail | 0.40 s sleep + 16 chunks | 3.656 | 2.159 | 1.331 | 0.176 | 0.152 | Roughly constant e: fixed Amdahl serialization |
| B: Lock Contention | shared Value + Lock | 4.063 | 11.963 | 11.989 | 4.842 | 3.540 | Severe lock thrashing: T2 and T4 are about 2.9x T1 |
| C: Privatized State | private accumulators, one reduction | 0.480 | 0.256 | 0.154 | 0.067 | 0.093 | Scaling restored: T4 is about 3.1x faster than T1 |

## Rubric self-check

| Check | Measured | Verdict |
|---|---|---|
| A: e(4)/e(2) in [0.80, 1.25] | 0.865 (runs: 0.800, 0.867, 0.927) | Met |
| B: T4 > T1 and e(4) >> e(2) | T4 = 11.989 s vs T1 = 4.063 s; e(4) = 3.540 < e(2) = 4.842 | Partly met: slowdown is clear, e(4) is not larger than e(2) |
| C: T4 < T1 and E4 > 70% | T4 = 0.154 s vs T1 = 0.480 s; E4 = 78% (runs: 76%, 78%, 80%) | Met |

## Notes

1. **Experiment A.** e stays flat between P=2 and P=4 (0.176 vs 0.152), which matches a fixed serial path. The 0.40 s sleep is about 11% of T1. The measured e is a little above the handout's 0.10–0.14 range, probably because starting the process pool and collecting results add extra serial time.
2. **Experiment B.** T2 and T4 are almost identical (about 12 s), so adding processes beyond two does not slow it down further. In the Karp–Flatt formula the denominator (p − 1) keeps growing while the runtime stays flat, so e(4) comes out smaller than e(2). The contention itself is unambiguous: the program is about 2.9x slower on 2 or 4 processes than on 1, and both e values are far above the 0.50 and 1.50 thresholds in the handout table.
3. **Experiment C.** Efficiency at P=4 is 78%. The measured e values (0.067 and 0.093) are above the handout's expected "< 0.05". The whole run lasts only about 0.15–0.5 s, so process-pool startup is a visible fixed cost. I did not verify this explanation separately.

## Conclusion

A fixed serial path (Experiment A) keeps e roughly constant across worker counts. A shared lock (Experiment B) makes the program slower with more processes, and the very high e shows contention rather than algorithmic serialization. Privatizing the counter (Experiment C) restores good scaling: about 3.1x speedup on 4 processes with 78% efficiency.
