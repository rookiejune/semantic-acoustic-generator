#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

cd "${SEMANTIC_ACOUSTIC_CODEC_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo '{"event":"job.launch","project":"semantic-acoustic-codec","experiment":"001","codec":"bicodec","route":"rvq"}'
"${SEMANTIC_ACOUSTIC_CODEC_PYTHON}" scripts/train.py \
    codec=bicodec \
    route=rvq \
    data.source=qwen_cross_text \
    decoder.rvq_predictor=mtp \
    output_subdir=bicodec/rvq-8l/fixed-speaker \
    "$@"
