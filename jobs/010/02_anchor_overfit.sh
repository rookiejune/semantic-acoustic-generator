#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

ANCHOR_SAMPLE_LIMIT="${ANCHOR_SAMPLE_LIMIT:-1}"
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-1}"
ANCHOR_MAX_STEPS="${ANCHOR_MAX_STEPS:-2000}"
ANCHOR_MODE="${ANCHOR_MODE:-anchor}"
ANCHOR_OUTPUT_SUBDIR="${ANCHOR_OUTPUT_SUBDIR:-010/${ANCHOR_MODE}-${ANCHOR_SAMPLE_LIMIT}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"010","stage":"anchor-overfit"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=010_longcat_anchor_overfit \
    datamodule.sample_limit="${ANCHOR_SAMPLE_LIMIT}" \
    datamodule.batch_size="${ANCHOR_BATCH_SIZE}" \
    model.decoder.fm_mode="${ANCHOR_MODE}" \
    trainer.max_steps="${ANCHOR_MAX_STEPS}" \
    output_subdir="${ANCHOR_OUTPUT_SUBDIR}" \
    "$@"
