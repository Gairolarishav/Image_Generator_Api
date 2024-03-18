#!/bin/bash

# Download and extract Tesseract OCR binaries
wget https://github.com/tesseract-ocr/tesseract/releases/download/v5.0.0/tesseract-5.0.0-linux-amd64.tar.xz
tar xf tesseract-5.0.0-linux-amd64.tar.xz

# Move Tesseract binaries to a location in the system PATH
sudo mv tesseract-5.0.0-linux-amd64/tesseract /usr/local/bin/
sudo mv tesseract-5.0.0-linux-amd64/tesseract.1 /usr/local/share/man/man1/

# Verify installation
tesseract --version

# install dependencies
pip install -r requirements.txt
