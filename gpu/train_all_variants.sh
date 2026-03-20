#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANTS=(
    baseline_ppo
    confidence_brier
    confidence_ce
    ppo_value
    confidence_prod
    confidence_min
)

if [ -n "${SEEDMIT_TRAIN_VARIANTS:-}" ]; then
    IFS=',' read -r -a VARIANTS <<< "${SEEDMIT_TRAIN_VARIANTS}"
fi

for variant in "${VARIANTS[@]}"; do
    bash "$SCRIPT_DIR/train_variant.sh" "$variant" "$@"
done
