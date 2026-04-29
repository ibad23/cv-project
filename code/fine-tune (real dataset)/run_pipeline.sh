#!/bin/bash

# Exit on error
set -e

echo "=== XLPSR Fine-Tuning Pipeline ==="

# 1. Generate Synthetic Data
echo "Step 1: Generating synthetic French plate dataset (50k)..."
python code/synth/gen_french_plates.py --n 50000 --out datasets/synth_french_plates

# 2. Fine-tune OCR
echo "Step 2: Fine-tuning fast-plate-ocr on synthetic data..."
# Note: Actual training might take 3-6 hours on T4
python code/finetune/train_fastplate.py --data datasets/synth_french_plates --epochs 10 --out models/fastplate_french_siv

# 3. Fine-tune SR
echo "Step 3: Fine-tuning Real-ESRGAN with OCR-aware loss (LCDNet-style)..."
# Note: Actual training might take 12-18 hours on T4
python code/finetune/train_realesrgan_lcd.py --data datasets/synth_french_plates --ocr_model models/fastplate_french_siv/model.pt --out models/realesrgan_x4_french_lcd

echo "=== Pipeline Complete! ==="
echo "New weights available at:"
echo "OCR: models/fastplate_french_siv/fastplate_french_siv.onnx"
echo "SR:  models/realesrgan_x4_french_lcd/realesrgan_x4_french_lcd.pth"
echo ""
echo "Next step: Update code/ultraplate_pipeline.ipynb model paths and run evaluation."
