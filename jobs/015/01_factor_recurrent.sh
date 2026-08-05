#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FACTOR_DATA_ROOT:?Set LONGCAT_FACTOR_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

FACTOR_RECURRENT_SAMPLE_LIMIT="${FACTOR_RECURRENT_SAMPLE_LIMIT:-10000}"
FACTOR_RECURRENT_BATCH_SIZE="${FACTOR_RECURRENT_BATCH_SIZE:-32}"
FACTOR_RECURRENT_MAX_STEPS="${FACTOR_RECURRENT_MAX_STEPS:-20000}"
FACTOR_RECURRENT_OUTPUT_SUBDIR="${FACTOR_RECURRENT_OUTPUT_SUBDIR:-015/depth-recurrent-2cb-${FACTOR_RECURRENT_SAMPLE_LIMIT}-${FACTOR_RECURRENT_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"015","stage":"factor-depth-recurrent-2cb"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
  experiment=015_longcat_factor_recurrent \
  datamodule.root="${LONGCAT_FACTOR_DATA_ROOT}" \
  datamodule.sample_limit="${FACTOR_RECURRENT_SAMPLE_LIMIT}" \
  datamodule.batch_size="${FACTOR_RECURRENT_BATCH_SIZE}" \
  trainer.max_steps="${FACTOR_RECURRENT_MAX_STEPS}" \
  output_subdir="${FACTOR_RECURRENT_OUTPUT_SUBDIR}" \
  "$@"
