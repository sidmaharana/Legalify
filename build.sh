#!/bin/bash
# Update packages
sudo apt-get update

# Install Poppler (for PDF rendering) and Tesseract OCR
sudo apt-get install -y poppler-utils tesseract-ocr

# Verify installations
echo "Poppler version:"
pdftoppm -v
echo "Tesseract version:"
tesseract --version

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
