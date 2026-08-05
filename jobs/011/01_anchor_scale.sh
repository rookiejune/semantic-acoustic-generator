#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_ANCHOR_SCALE_DATA_ROOT:?Set LONGCAT_ANCHOR_SCALE_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

ANCHOR_SCALE_SAMPLE_LIMIT="${ANCHOR_SCALE_SAMPLE_LIMIT:-10000}"
ANCHOR_SCALE_BATCH_SIZE="${ANCHOR_SCALE_BATCH_SIZE:-16}"
ANCHOR_SCALE_MAX_STEPS="${ANCHOR_SCALE_MAX_STEPS:-2000}"
ANCHOR_SCALE_OUTPUT_SUBDIR="${ANCHOR_SCALE_OUTPUT_SUBDIR:-011/local-anchor-${ANCHOR_SCALE_SAMPLE_LIMIT}-${ANCHOR_SCALE_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"011","stage":"anchor-scale"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=011_longcat_anchor_scale \
    datamodule.root="${LONGCAT_ANCHOR_SCALE_DATA_ROOT}" \
    datamodule.sample_limit="${ANCHOR_SCALE_SAMPLE_LIMIT}" \
    datamodule.batch_size="${ANCHOR_SCALE_BATCH_SIZE}" \
    trainer.max_steps="${ANCHOR_SCALE_MAX_STEPS}" \
    output_subdir="${ANCHOR_SCALE_OUTPUT_SUBDIR}" \
    "$@"
