#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Skipping Poppler and Tesseract Installation (OCR removed) ---"

# Continue with Python dependencies
echo "--- Installing Python Dependencies ---"
pip install -r backend/requirements.txt
echo "--- Python Dependencies Installation Complete ---"