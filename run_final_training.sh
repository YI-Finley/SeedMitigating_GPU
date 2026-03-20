#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$PROJECT_ROOT/run_full_training.sh" "$@"
