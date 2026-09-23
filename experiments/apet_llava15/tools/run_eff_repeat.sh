#!/usr/bin/env bash
set -euo pipefail

REP="${1:?Usage: run_eff_repeat.sh r2|r3 GPU_ID}"
GPU="${2:?Usage: run_eff_repeat.sh r2|r3 GPU_ID}"

if [[ "$REP" != "r2" && "$REP" != "r3" ]]; then
    echo "REP must be r2 or r3"
    exit 1
fi

APET_ROOT="$(pwd)"
EXP="$APET_ROOT/experiments/apet_llava15"

QUESTION_FILE="data/eval/pope/efficiency/pope_eff_520_s42.jsonl"
IMAGE_FOLDER="data/eval/pope/val2014"
MODEL_PATH="data/model/llava-v1.5-7b"

mkdir -p "$EXP/efficiency/raw"
mkdir -p "$EXP/efficiency/logs"

echo "======================================================"
echo "Efficiency benchmark"
echo "Repeat: $REP"
echo "Physical GPU: $GPU"
echo "Date: $(date)"
echo "Project: $APET_ROOT"
echo "======================================================"

nvidia-smi -i "$GPU" \
  > "$EXP/efficiency/logs/nvidia_smi_${REP}_before.txt"

run_one () {
    RUN_ID="$1"
    shift

    echo
    echo "======================================================"
    echo "START: $RUN_ID"
    echo "TIME : $(date)"
    echo "GPU  : $GPU"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES="$GPU" \
    python "$EXP/tools/benchmark_efficiency.py" \
      --run-id "$RUN_ID" \
      --model-path "$MODEL_PATH" \
      --question-file "$QUESTION_FILE" \
      --image-folder "$IMAGE_FOLDER" \
      --output-prefix "$EXP/efficiency/raw/$RUN_ID" \
      --conv-mode vicuna_v1 \
      --basis-token-num 10 \
      --warmup 20 \
      --measure 500 \
      --max-new-tokens 128 \
      --seed 42 \
      "$@" \
      2>&1 | tee "$EXP/efficiency/logs/${RUN_ID}.log"

    echo
    echo "FINISHED: $RUN_ID"
    echo "TIME    : $(date)"
    echo

    # 给GPU一点时间回到比较稳定的状态
    sleep 15
}

# ------------------------------------------------------
# 1. Vanilla-576
# ------------------------------------------------------
run_one "eff_vanilla_576_${REP}" \
    --visual-token-num 576 \
    --layer-list 'None' \
    --image-token-list 'None'

# ------------------------------------------------------
# 2. ApET-192: 288 -> 96
# ------------------------------------------------------
run_one "eff_apet_192_${REP}" \
    --visual-token-num 288 \
    --layer-list '[16]' \
    --image-token-list '[96]'

# ------------------------------------------------------
# 3. ApET-128: 192 -> 64
# ------------------------------------------------------
run_one "eff_apet_128_${REP}" \
    --visual-token-num 192 \
    --layer-list '[16]' \
    --image-token-list '[64]'

# ------------------------------------------------------
# 4. ApET-64: 96 -> 32
# ------------------------------------------------------
run_one "eff_apet_64_${REP}" \
    --visual-token-num 96 \
    --layer-list '[16]' \
    --image-token-list '[32]'

nvidia-smi -i "$GPU" \
  > "$EXP/efficiency/logs/nvidia_smi_${REP}_after.txt"

echo
echo "======================================================"
echo "$REP ALL FINISHED"
echo "TIME: $(date)"
echo "======================================================"
