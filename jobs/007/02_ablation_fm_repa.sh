#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

cd "${SEMANTIC_ACOUSTIC_GENERATOR_ROOT}"
export PYTHONUNBUFFERED=1

echo '{"event":"job.launch","project":"semantic-acoustic-generator","experiment":"007","cell":"A1","route":"fm"}'
"${SEMANTIC_ACOUSTIC_GENERATOR_PYTHON}" scripts/train.py \
    experiment=ablation_fm_repa \
    "$@"
