#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL.gguf [llama-bench options...]" >&2
    exit 2
fi

model=$1
shift
prompt_tokens=${PROMPT_TOKENS:-512}
generation_tokens=${GENERATION_TOKENS:-128}
repetitions=${REPETITIONS:-5}
batch_size=${BATCH_SIZE:-2048}
ubatch_size=${UBATCH_SIZE:-512}
n_gpu_layers=${N_GPU_LAYERS:--1}

args=(
    --model "$model"
    --pg "${prompt_tokens},${generation_tokens}"
    --repetitions "$repetitions"
    --batch-size "$batch_size"
    --ubatch-size "$ubatch_size"
    --n-gpu-layers "$n_gpu_layers"
    --output json
)

if [[ -n "${THREADS:-}" ]]; then
    args+=(--threads "$THREADS")
fi

exec llama-bench "${args[@]}" "$@"
