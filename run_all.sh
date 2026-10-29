#!/usr/bin/env bash
# Runs the measurement scripts in lab order. Usage: ./run_all.sh <8-digit-student-id>
# It does NOT run git or verify_outputs.py: commit + tag after each milestone yourself
# (timestamps must fall inside the 180-minute window).
set -euo pipefail
cd "$(dirname "$0")"
ID="${1:?usage: ./run_all.sh <student-id>}"

python3 src/profile_env.py --student-id "$ID"          # M0  -> git tag M0-ENV
python3 src/stride_bench.py                            # M1
python3 src/dense_kernel.py                            # M1  -> git tag M1-LOCALITY
python3 src/false_sharing.py                           # M2  -> git tag M2-SHARING
python3 src/scaling_study.py                           # M3  -> git tag M3-SCALING
echo "Now: python3 verify_outputs.py ; write README.md ; git tag M4-FINAL ; git push origin main --tags"
