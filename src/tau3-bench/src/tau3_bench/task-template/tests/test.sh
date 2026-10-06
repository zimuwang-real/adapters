#!/bin/bash
set -euo pipefail

LOG_DIR="/logs/verifier"
mkdir -p "${LOG_DIR}"

python3 /tests/evaluate.py \
  --evaluation /tmp/tau3-evaluation.json \
  --reward "${LOG_DIR}/reward.txt" \
  --result "${LOG_DIR}/result.json"
