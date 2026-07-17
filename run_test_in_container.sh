#!/bin/bash
set -euo pipefail
TEST_FILE="$1"
cd /root/vime
pip install -e . --no-deps --break-system-packages 2>&1 | tail -2
python3 "tests/$TEST_FILE"
