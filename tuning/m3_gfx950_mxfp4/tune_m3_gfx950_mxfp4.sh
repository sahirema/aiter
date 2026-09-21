#!/usr/bin/env bash
# Tune the MiniMax-M3-MXFP4 dense MXFP4 GEMM shapes on gfx950 / MI355X.
#
# Produces measured JSON configs for the GEMM-AFP4WFP4 (plain, shuffle=False)
# family -- the family sglang's quark W4A4 MXFP4 linear actually consults
# (quark_w4a4_mxfp4.py:341 -> aiter gemm_afp4wfp4 -> _get_config(shuffle=False)).
#
# This script MUST run on a GPU node. It never writes into the aiter config
# tree; installing the generated JSONs is a separate, reviewed step (see README).
#
# rc=0 is NOT evidence that anything ran. Check the log for the sentinel
#   "=== M3 TUNE COMPLETE ==="
# and for one "screen done" line per (M,N,K) case.

set -euo pipefail

AITER_ROOT="${AITER_ROOT:-$HOME/aiter}"
TUNE_DIR="$AITER_ROOT/aiter/ops/triton/utils/_triton/tunning"
OUT_DIR="${OUT_DIR:-$AITER_ROOT/tuning/m3_gfx950_mxfp4/out}"
LOG="${LOG:-$OUT_DIR/tune_m3_gfx950_mxfp4.log}"
NUM_GPUS="${NUM_GPUS:-8}"

# M buckets == STANDARD_M_BOUNDS in gemm_config_utils.py:25, so view-screen.py
# can emit one M_LEQ_<bound> key per bucket with no gaps.
M_LIST="${M_LIST:-1 4 8 16 32 64 128 256 512 1024 2048 4096 8192}"

# (N:K) pairs from shapes_m3_mxfp4.csv. K is the LOGICAL (unpacked) K.
# Six tp_sharded shapes (tp2/tp4/tp8 x gate_up/down) plus the replicated
# (DeepEP, tp_size=1) pair, which is TP-independent and so appears once.
# Which variant is live depends on the a2a backend; which tp_sharded pair is
# live depends on --tp. Sweeping all eight costs tuning time only -- an unused
# JSON is never loaded, since the lookup is keyed on (N,K).
#
# Override for a single TP degree, e.g. tp2 plus the replicated pair:
#   SHAPES="3072:6144 6144:1536 6144:6144 6144:3072" ./tune_m3_gfx950_mxfp4.sh
SHAPES="${SHAPES:-3072:6144 6144:1536 1536:6144 6144:768 768:6144 6144:384 6144:6144 6144:3072}"

mkdir -p "$OUT_DIR"
# Timestamp marker: the log is appended to throughout the run, so it can never
# be used as the -newer reference for collecting output JSONs.
STAMP="$OUT_DIR/.run_started"
: > "$STAMP"
cd "$TUNE_DIR"

{
  echo "=== M3 TUNE START $(date -Is) ==="
  echo "host=$(hostname)"
  echo "aiter_root=$AITER_ROOT"
  echo "aiter_git=$(git -C "$AITER_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(python3 -c 'import sys;print(sys.version.split()[0])')"
  echo "torch=$(python3 -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"
  echo "arch=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName)' 2>&1 | tail -1)"
  echo "M_LIST=$M_LIST"
  echo "SHAPES=$SHAPES"
} | tee -a "$LOG"

# gfx950 AFP4WFP4 kernel constraints, documented in tunning/README.md:
#   BLOCK_SIZE_K >= 256 always; BLOCK_SIZE_M < 32 for M < 32, >= 32 for M >= 32.
# Passing the right BLOCK_SIZE_M range per M skips the assert-and-prune pass that
# screen.py would otherwise spend minutes on.
gpu=0
for shape in $SHAPES; do
  N="${shape%%:*}"
  K="${shape##*:}"
  for M in $M_LIST; do
    if [ "$M" -lt 32 ]; then
      BSM_RANGE="16"
    else
      BSM_RANGE="32 64 128 256"
    fi
    echo "--- screen start M=$M N=$N K=$K gpu=$gpu ---" | tee -a "$LOG"
    # screen.py positional contract (screen.py:24-28): M N K G F
    python3 screen.py \
      "$M" "$N" "$K" "$gpu" \
      ut_afp4wfp4_gemm.py \
      --block-size-m-range $BSM_RANGE \
      --block-size-k-range 256 512 1024 \
      --overwrite \
      >> "$LOG" 2>&1
    echo "screen done M=$M N=$N K=$K rc=$?" | tee -a "$LOG"
    gpu=$(( (gpu + 1) % NUM_GPUS ))
  done
done

# view-screen.py turns the per-M screen*.log files into one JSON per (N,K),
# named <config_name>-N=<N>-K=<K>.json. Emitted into $OUT_DIR, NOT installed.
for shape in $SHAPES; do
  N="${shape%%:*}"
  K="${shape##*:}"
  echo "--- view-screen N=$N K=$K ---" | tee -a "$LOG"
  # --json-prefix is mandatory here. Left to guess, view-screen.py:70 builds
  # "{DEVICE_ARCH}-GEMM-AFP4WFP4", i.e. the OLD arch-prefixed flat name. The
  # current nested layout (configs/<arch>/<backend>/gemm/<d_type>/) carries NO
  # arch prefix, so a guessed name would sit in the tree and never be loaded --
  # get_gemm_config would silently keep returning DEFAULT.json.
  python3 view-screen.py ut_afp4wfp4_gemm.py \
    --n-list "$N" --k-list "$K" \
    --json-prefix GEMM-AFP4WFP4 >> "$LOG" 2>&1
done

# Collect whatever JSON the tuner produced next to the UT into OUT_DIR.
find "$TUNE_DIR" -maxdepth 1 -name 'GEMM-AFP4WFP4-N=*.json' -newer "$STAMP" -print \
  -exec cp {} "$OUT_DIR"/ \; | tee -a "$LOG" || true

{
  echo "generated JSON files in $OUT_DIR:"
  ls -l "$OUT_DIR"/GEMM-AFP4WFP4-N=*.json 2>/dev/null || echo "NONE -- tuning produced nothing, do NOT install"
  echo "=== M3 TUNE COMPLETE $(date -Is) ==="
} | tee -a "$LOG"
