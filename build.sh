#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing Poppler ---"
# Install Poppler utilities (poppler-utils)
# This command is for Debian/Ubuntu based systems, which Render uses.
apt-get update && apt-get install -y poppler-utils

echo "--- Poppler Installation Complete ---"

# Continue with Python dependencies
echo "--- Installing Python Dependencies ---"
pip install -r backend/requirements.txt
echo "--- Python Dependencies Installation Complete ---"