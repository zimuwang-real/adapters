#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier

python3 /tests/evaluate.py \
  --evaluation /tmp/tau3-evaluation.json \
  --reward /logs/verifier/reward.txt \
  --result /logs/verifier/result.json
