#!/usr/bin/env bash
set -euo pipefail

REP="${1:?Usage: run_randomprune_eff_repeat.sh r1|r2|r3 GPU_ID}"
GPU="${2:?Usage: run_randomprune_eff_repeat.sh r1|r2|r3 GPU_ID}"

if [[ "$REP" != "r1" && "$REP" != "r2" && "$REP" != "r3" ]]; then
    echo "REP must be r1, r2, or r3"
    exit 1
fi

APET_ROOT="$(pwd)"
EXP="$APET_ROOT/experiments/apet_llava15"

MODEL_PATH="data/model/llava-v1.5-7b"
QUESTION_FILE="data/eval/pope/efficiency/pope_eff_520_s42.jsonl"
IMAGE_FOLDER="data/eval/pope/val2014"

RUN_ID="eff_randomprune_64_${REP}"

mkdir -p "$EXP/random_prune/efficiency/raw"
mkdir -p "$EXP/random_prune/efficiency/logs"

echo "=================================================="
echo "RandomPrune-64 efficiency benchmark"
echo "Repeat       : $REP"
echo "Physical GPU : $GPU"
echo "Start        : $(date)"
echo "=================================================="

nvidia-smi -i "$GPU" \
> "$EXP/random_prune/efficiency/logs/nvidia_smi_${REP}_before.txt"

CUDA_VISIBLE_DEVICES="$GPU" \
python "$EXP/tools/benchmark_efficiency.py" \
  --run-id "$RUN_ID" \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --output-prefix "$EXP/random_prune/efficiency/raw/$RUN_ID" \
  --conv-mode vicuna_v1 \
  --visual-token-num 96 \
  --layer-list '[16]' \
  --image-token-list '[32]' \
  --basis-token-num 10 \
  --token-selection-method random_prune \
  --warmup 20 \
  --measure 500 \
  --max-new-tokens 128 \
  --seed 42 \
  2>&1 | tee \
  "$EXP/random_prune/efficiency/logs/${RUN_ID}.log"

nvidia-smi -i "$GPU" \
> "$EXP/random_prune/efficiency/logs/nvidia_smi_${REP}_after.txt"

echo
echo "=================================================="
echo "$RUN_ID FINISHED"
echo "End: $(date)"
echo "=================================================="
