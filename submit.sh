#!/usr/bin/env bash
# Commit + tag each milestone in order, run verification, final commit/tag, optional push.
#   ./submit.sh            -> local commits/tags only
#   ./submit.sh --push     -> also: git push origin main --tags
# Tags get the real current time; nothing is back-dated.
set -euo pipefail
cd "$(dirname "$0")"

[ -d .git ] || { echo "No git repo here. Clone/init the course repo first (git init -b main; git remote add origin <url>)."; exit 1; }

milestone() {  # tag, message, files...
  local tag="$1" msg="$2"; shift 2
  git add -- "$@" 2>/dev/null || git add -A -- "$@"
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "tag $tag already exists - skipping"; return
  fi
  git diff --cached --quiet || git commit -m "$msg"
  git tag "$tag"
  echo "tagged $tag"
}

milestone M0-ENV "M0: Environment and salting configuration" \
  src/profile_env.py src/salt.py results/host_profile.json results/salted_params.json
milestone M1-LOCALITY "M1: Memory stride and loop interchange benchmarking" \
  src/stride_bench.py src/dense_kernel.py results/stride_results.tsv results/loop_interchange.json
milestone M2-SHARING "M2: False sharing reproduction and Karp-Flatt analysis" \
  src/false_sharing.py results/false_sharing_karp_flatt.tsv
milestone M3-SCALING "M3: Strong and weak scaling evaluation" \
  src/scaling_study.py results/scaling_metrics.tsv results/roofline_summary.json results/scaling_fit.json

if [ -f verify_outputs.py ]; then
  python3 verify_outputs.py
else
  echo "verify_outputs.py not found in $(pwd): copy it here and re-run, or run it from where you got it." >&2
  exit 1
fi

python3 make_readme.py >/dev/null
git add .
git diff --cached --quiet || git commit -m "M4: Final verification and completed practicum submission"
git rev-parse -q --verify refs/tags/M4-FINAL >/dev/null || git tag M4-FINAL
echo "tagged M4-FINAL"

if [ "${1:-}" = "--push" ]; then
  git push origin main --tags
else
  echo "Review with: git log --oneline --decorate   then:  git push origin main --tags"
fi
