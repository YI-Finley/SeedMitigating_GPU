#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_FILE="$PROJECT_ROOT/data/dapo_math_train_1k.parquet"
VAL_FILE="$PROJECT_ROOT/data/dapo_math_val.parquet"

bash "$PROJECT_ROOT/gpu/train_all_variants.sh" \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${VAL_FILE}']" \
    "$@"
