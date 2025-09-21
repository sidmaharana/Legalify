#!/bin/bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr

# Optional: confirm installation
poppler_path=/usr/bin
echo "Poppler installed at $poppler_path"
tesseract --version

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
