#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

cd "${SEMANTIC_ACOUSTIC_CODEC_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo '{"event":"job.launch","project":"semantic-acoustic-codec","experiment":"001","codec":"longcat","route":"dit"}'
"${SEMANTIC_ACOUSTIC_CODEC_PYTHON}" scripts/train.py \
    experiment=001_longcat_fm \
    "$@"
