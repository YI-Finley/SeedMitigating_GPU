#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/eval_response_level_beyondaime.sh"
bash "$SCRIPT_DIR/eval_response_level_aime.sh"
bash "$SCRIPT_DIR/eval_simpleqa.sh"
bash "$SCRIPT_DIR/eval_claim_level_beyondaime.sh"
