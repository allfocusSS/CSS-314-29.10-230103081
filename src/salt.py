"""Deterministic Student-ID salting (Milestone 0, Task 0.2)."""
import hashlib, os
import json


def get_student_params(student_id: str):
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)
    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216
    # Seed for deterministic random number generation
    seed = h % 100_007
    return {"student_id": student_id, "N": N, "buffer_elements": buffer_elements, "seed": seed}


def load_params(results_dir, student_id=None):
    """Return salted params: recompute from --student-id if given, else read
    results/salted_params.json written by profile_env.py."""
    if student_id:
        return get_student_params(student_id)
    path = os.path.join(results_dir, "salted_params.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run  python3 src/profile_env.py --student-id <ID>  first "
                         "(or pass --student-id).")
    with open(path) as f:
        return json.load(f)
