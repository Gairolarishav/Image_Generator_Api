# Use a base image with the necessary environment (e.g., Debian or Alpine)
FROM python:3.9-slim-buster

# Update package lists and install Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory
WORKDIR /image_generator_api

COPY main.py /image_generator_api/main.py

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
