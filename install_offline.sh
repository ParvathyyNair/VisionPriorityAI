#!/usr/bin/env sh
# Installs all dependencies from ./wheelhouse - no internet needed.
cd "$(dirname "$0")" && python3 -m pip install --no-index --find-links wheelhouse -r requirements.txt && python3 scripts/offline_check.py
