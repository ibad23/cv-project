#!/bin/bash

# Exit on error
set -e

echo "Starting project setup..."

# 1. Create directories
mkdir -p datasets/fonts
mkdir -p external
mkdir -p models

# 2. Clone repositories
echo "Cloning external repositories..."
if [ ! -d "external/Real-ESRGAN" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git external/Real-ESRGAN
fi

if [ ! -d "external/LCDNet" ]; then
    git clone https://github.com/valfride/lpsr-lacd.git external/LCDNet
fi

# 3. Download Fonts
echo "Downloading fonts..."
if [ ! -f "datasets/fonts/CharlesWright.ttf" ]; then
    curl -L "https://github.com/the-muda-organization/charles-wright-font/raw/master/Charles%20Wright%202001.ttf" -o "datasets/fonts/CharlesWright.ttf"
fi

# 4. Download datasets using the python script
echo "Downloading datasets..."
python code/download_datasets.py

echo "Setup complete! Please update your conda environment using:"
echo "conda env update -f environment.yml"
