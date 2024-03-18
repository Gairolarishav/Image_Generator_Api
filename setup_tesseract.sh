#!/bin/bash
tar xf /5.3.4.tar.gz

# Move Tesseract binaries to a location in the system PATH
sudo mv tesseract-5.0.0-linux-amd64/tesseract /usr/local/bin/
sudo mv tesseract-5.0.0-linux-amd64/tesseract.1 /usr/local/share/man/man1/

# Verify installation
tesseract --version

sudo apt install tesseract-ocr 

# install dependencies
pip install -r requirements.txt
