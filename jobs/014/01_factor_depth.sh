#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FACTOR_DATA_ROOT:?Set LONGCAT_FACTOR_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

FACTOR_DEPTH_SAMPLE_LIMIT="${FACTOR_DEPTH_SAMPLE_LIMIT:-10000}"
FACTOR_DEPTH_BATCH_SIZE="${FACTOR_DEPTH_BATCH_SIZE:-16}"
FACTOR_DEPTH_MAX_STEPS="${FACTOR_DEPTH_MAX_STEPS:-2000}"
FACTOR_DEPTH_OUTPUT_SUBDIR="${FACTOR_DEPTH_OUTPUT_SUBDIR:-014/depth-ar-2cb-${FACTOR_DEPTH_SAMPLE_LIMIT}-${FACTOR_DEPTH_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"014","stage":"factor-depth-ar-2cb"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=014_longcat_factor_depth \
    datamodule.root="${LONGCAT_FACTOR_DATA_ROOT}" \
    datamodule.sample_limit="${FACTOR_DEPTH_SAMPLE_LIMIT}" \
    datamodule.batch_size="${FACTOR_DEPTH_BATCH_SIZE}" \
    trainer.max_steps="${FACTOR_DEPTH_MAX_STEPS}" \
    output_subdir="${FACTOR_DEPTH_OUTPUT_SUBDIR}" \
    "$@"
