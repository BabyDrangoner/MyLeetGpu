#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --user pip-tools
python3 -m piptools compile --generate-hashes --resolver=backtracking --output-file requirements.lock requirements.in

