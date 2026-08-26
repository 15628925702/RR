#!/usr/bin/env bash
set -u
ROOT=/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/current_20260825
ENV=/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/env/bin/python
LOG=$ROOT/p4_formal_rejection_20260826.log
OUT=$ROOT/results/p4_formal_rejection_20260826
LOGDIR=$ROOT/p4_formal_rejection_20260826_logs
PREPARED=$ROOT/experiments/p4_prepared_oracle_hp.pkl
mkdir -p "$OUT" "$LOGDIR"
cd "$ROOT" || exit 1
echo "START $(date -Is)" >> "$LOG"
echo "PWD $(pwd)" >> "$LOG"
echo "PYTHON $ENV" >> "$LOG"
$ENV - "$PREPARED" <<'PY' >> "$LOG" 2>&1
import hashlib, pickle, sys
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
payload = pickle.load(open(path, "rb"))
reference = len(payload.get("reference", []))
large = len(payload.get("reference_large", []))
shape = tuple(getattr(payload.get("information", None), "shape", ()))
print("PREPARED", path)
print("SHA256", digest.hexdigest())
print("reference", reference)
print("reference_large", large)
print("information_shape", shape)
if reference < 50000 or large < 1000000 or shape != (120, 12, 12):
    raise SystemExit("prepared artifact failed formal size gate")
PY
status=$?
if [ "$status" != "0" ]; then
  echo "PREPARED_FAIL $(date -Is)" >> "$LOG"
  exit 1
fi
echo "PREPARED_READY $(date -Is)" >> "$LOG"

run_budget() {
  local B="$1"
  local extra="$2"
  local tag="$3"
  echo "B_START $tag $B $(date -Is)" >> "$LOG"
  pids=()
  for i in $(seq 0 7); do
    a=$((i * 25))
    b=$(((i + 1) * 25))
    mid=$((a + 12))
    for part in 0 1; do
      if [ "$part" = 0 ]; then s=$a; e=$mid; else s=$mid; e=$b; fi
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      CUDA_VISIBLE_DEVICES=$i RR_GID_CN_CUDA_DEVICE=0 PYTHONPATH=$ROOT/src PYTHONUNBUFFERED=1 \
        setsid nohup "$ENV" "$ROOT/scripts/p4_formal_run.py" \
          --config "$ROOT/configs/p4_formal.yaml" \
          --budget "$B" \
          --rep-range "$s" "$e" \
          --prepared "$PREPARED" \
          --out-prefix "p4_formal_rejection_20260826/gpu${i}p${part}${tag}" \
          $extra \
          > "$LOGDIR/B${B}${tag}_gpu${i}p${part}.log" 2>&1 < /dev/null &
      pids+=("$!")
    done
  done
  fail=0
  for p in "${pids[@]}"; do
    wait "$p" || fail=1
  done
  echo "B_DONE $tag $B fail=$fail $(date -Is)" >> "$LOG"
  if [ "$fail" = "1" ]; then
    echo "STOP_AFTER_FAIL $tag $B" >> "$LOG"
    exit 1
  fi
}

for B in 2000 4000 8000 16000 32000; do
  run_budget "$B" "" ""
done
for J in 0 1; do
  run_budget 8000 "--scoring-steps $J" "_J${J}"
done
echo "ALL_DONE $(date -Is)" >> "$LOG"
