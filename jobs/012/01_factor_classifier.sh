#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FACTOR_DATA_ROOT:?Set LONGCAT_FACTOR_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

FACTOR_SAMPLE_LIMIT="${FACTOR_SAMPLE_LIMIT:-128}"
FACTOR_BATCH_SIZE="${FACTOR_BATCH_SIZE:-16}"
FACTOR_MAX_STEPS="${FACTOR_MAX_STEPS:-2000}"
FACTOR_OUTPUT_SUBDIR="${FACTOR_OUTPUT_SUBDIR:-012/factor-anchor-${FACTOR_SAMPLE_LIMIT}-${FACTOR_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"012","stage":"factor-classifier"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=012_longcat_factor_classifier \
    datamodule.root="${LONGCAT_FACTOR_DATA_ROOT}" \
    datamodule.sample_limit="${FACTOR_SAMPLE_LIMIT}" \
    datamodule.batch_size="${FACTOR_BATCH_SIZE}" \
    trainer.max_steps="${FACTOR_MAX_STEPS}" \
    output_subdir="${FACTOR_OUTPUT_SUBDIR}" \
    "$@"
