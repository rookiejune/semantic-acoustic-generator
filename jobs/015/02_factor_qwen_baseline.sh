#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FACTOR_DATA_ROOT:?Set LONGCAT_FACTOR_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export PYTHONUNBUFFERED=1

FACTOR_QWEN_SAMPLE_LIMIT="${FACTOR_QWEN_SAMPLE_LIMIT:-9984}"
FACTOR_QWEN_BATCH_SIZE="${FACTOR_QWEN_BATCH_SIZE:-96}"
FACTOR_QWEN_MAX_STEPS="${FACTOR_QWEN_MAX_STEPS:-20000}"
FACTOR_QWEN_OUTPUT_SUBDIR="${FACTOR_QWEN_OUTPUT_SUBDIR:-015/depth-qwen-original-2cb-${FACTOR_QWEN_SAMPLE_LIMIT}-${FACTOR_QWEN_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"015","stage":"factor-depth-qwen-original-2cb"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
  experiment=015_longcat_factor_recurrent \
  datamodule.root="${LONGCAT_FACTOR_DATA_ROOT}" \
  datamodule.sample_limit="${FACTOR_QWEN_SAMPLE_LIMIT}" \
  datamodule.batch_size="${FACTOR_QWEN_BATCH_SIZE}" \
  model.decoder.factor_predictor=depth_ar \
  pl_module.residual_retarget=false \
  trainer.max_steps="${FACTOR_QWEN_MAX_STEPS}" \
  output_subdir="${FACTOR_QWEN_OUTPUT_SUBDIR}" \
  "$@"
