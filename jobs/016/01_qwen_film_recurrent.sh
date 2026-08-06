#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FACTOR_DATA_ROOT:?Set LONGCAT_FACTOR_DATA_ROOT}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

QWEN_FILM_SAMPLE_LIMIT="${QWEN_FILM_SAMPLE_LIMIT:-9984}"
QWEN_FILM_BATCH_SIZE="${QWEN_FILM_BATCH_SIZE:-96}"
QWEN_FILM_MAX_STEPS="${QWEN_FILM_MAX_STEPS:-5000}"
QWEN_FILM_OUTPUT_SUBDIR="${QWEN_FILM_OUTPUT_SUBDIR:-016/qwen-film-time-recurrent-depth-2cb-${QWEN_FILM_SAMPLE_LIMIT}-${QWEN_FILM_MAX_STEPS}}"

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"016","stage":"qwen-film-time-recurrent-depth-2cb"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
  experiment=016_longcat_qwen_film_anchor \
  datamodule.root="${LONGCAT_FACTOR_DATA_ROOT}" \
  datamodule.sample_limit="${QWEN_FILM_SAMPLE_LIMIT}" \
  datamodule.batch_size="${QWEN_FILM_BATCH_SIZE}" \
  trainer.max_steps="${QWEN_FILM_MAX_STEPS}" \
  output_subdir="${QWEN_FILM_OUTPUT_SUBDIR}" \
  "$@"
