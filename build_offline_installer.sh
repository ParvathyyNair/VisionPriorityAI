#!/usr/bin/env sh
# Run ONCE with internet (same OS + Python version as the target machine).
cd "$(dirname "$0")" && python3 -m pip download -r requirements.txt -d wheelhouse
echo "Done. Copy the folder (with wheelhouse/) to the offline machine and run ./install_offline.sh"
