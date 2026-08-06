#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

: "${LONGCAT_FIRST_CODEBOOK_DATA_ROOT:?Set LONGCAT_FIRST_CODEBOOK_DATA_ROOT}"
: "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR:?Set LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export PYTHONUNBUFFERED=1

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"010","stage":"representation-oracle"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/eval_first_codebook_oracle.py \
    --data-root "${LONGCAT_FIRST_CODEBOOK_DATA_ROOT}" \
    --output-dir "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/oracle" \
    --device cuda \
    "$@"

"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/eval_speech_manifest.py \
    --manifest "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/oracle/manifest.private.json" \
    --output-dir "${LONGCAT_FIRST_CODEBOOK_OUTPUT_DIR}/oracle/anytrain-eval" \
    --device cuda \
    --whisper-root "${ANYTRAIN_WHISPER_ROOT:-${HF_HOME}/whisper}" \
    --allow-utmos-remote-code
