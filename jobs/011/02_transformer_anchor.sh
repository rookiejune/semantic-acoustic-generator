#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_ANCHOR_SCALE_DATA_ROOT:?Set LONGCAT_ANCHOR_SCALE_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

TRANSFORMER_ANCHOR_SAMPLE_LIMIT="${TRANSFORMER_ANCHOR_SAMPLE_LIMIT:-10000}"
TRANSFORMER_ANCHOR_BATCH_SIZE="${TRANSFORMER_ANCHOR_BATCH_SIZE:-16}"
TRANSFORMER_ANCHOR_MAX_STEPS="${TRANSFORMER_ANCHOR_MAX_STEPS:-2000}"
TRANSFORMER_ANCHOR_OUTPUT_SUBDIR="${TRANSFORMER_ANCHOR_OUTPUT_SUBDIR:-011/transformer-anchor-${TRANSFORMER_ANCHOR_SAMPLE_LIMIT}-${TRANSFORMER_ANCHOR_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"011","stage":"transformer-anchor"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=011_longcat_transformer_anchor \
    datamodule.root="${LONGCAT_ANCHOR_SCALE_DATA_ROOT}" \
    datamodule.sample_limit="${TRANSFORMER_ANCHOR_SAMPLE_LIMIT}" \
    datamodule.batch_size="${TRANSFORMER_ANCHOR_BATCH_SIZE}" \
    trainer.max_steps="${TRANSFORMER_ANCHOR_MAX_STEPS}" \
    output_subdir="${TRANSFORMER_ANCHOR_OUTPUT_SUBDIR}" \
    "$@"
