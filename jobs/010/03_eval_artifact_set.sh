#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FIRST_CODEBOOK_DATA_ROOT:?Set LONGCAT_FIRST_CODEBOOK_DATA_ROOT}"
: "${LONGCAT_FIRST_CODEBOOK_ARTIFACT:?Set LONGCAT_FIRST_CODEBOOK_ARTIFACT}"
: "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR:?Set LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"010","stage":"artifact-fixed-eval"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/eval_artifact_set.py \
    --artifact "${LONGCAT_FIRST_CODEBOOK_ARTIFACT}" \
    --data-root "${LONGCAT_FIRST_CODEBOOK_DATA_ROOT}" \
    --output-dir "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/artifact-eval" \
    --device cuda \
    "$@"

"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/eval_speech_manifest.py \
    --manifest "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/artifact-eval/manifest.private.json" \
    --output-dir "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/artifact-eval/anytrain-eval" \
    --device cuda \
    --whisper-root "${ANYTRAIN_WHISPER_ROOT:-${HF_HOME}/whisper}"
