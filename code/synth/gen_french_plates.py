import os
import random
import string
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

# Degradation utilities (Simplified version of Real-ESRGAN second-order)
def add_noise(img, sigma):
    noise = np.random.randn(*img.shape) * sigma
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_blur(img, ksize):
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def degrade_image(img_np, target_width):
    h, w = img_np.shape[:2]
    # Randomly choose degradation parameters
    # Blur -> Noise -> JPEG -> Downsample -> Blur -> Noise -> JPEG -> Upsample
    
    # 1. Blur
    img = apply_blur(img_np, random.randint(1, 3))
    # 2. Noise
    img = add_noise(img, random.uniform(0, 5))
    # 3. Downsample to target width
    scale = target_width / w
    target_height = int(h * scale)
    img_lr = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    # 4. More degradation on LR
    img_lr = apply_blur(img_lr, random.randint(1, 3))
    img_lr = add_noise(img_lr, random.uniform(0, 10))
    # 5. JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 90)]
    _, encimg = cv2.imencode('.jpg', img_lr, encode_param)
    img_lr = cv2.imdecode(encimg, 1)
    
    # 6. Upsample back to original size for SR training (optional, but often used)
    img_sr_input = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return img_lr, img_sr_input

class FrenchPlateGenerator:
    def __init__(self, font_path):
        self.font_path = font_path
        # Standard French SIV size (approx ratio 52:11)
        self.width = 520
        self.height = 110
        self.font_size = 80
        try:
            self.font = ImageFont.truetype(font_path, self.font_size)
        except:
            print(f"Font not found at {font_path}, using default.")
            self.font = ImageFont.load_default()

    def generate_text(self):
        # SIV: AA-NNN-AA
        letters = "".join(random.choices(string.ascii_uppercase, k=2))
        numbers = "".join(random.choices(string.digits, k=3))
        letters2 = "".join(random.choices(string.ascii_uppercase, k=2))
        return f"{letters}-{numbers}-{letters2}"

    def generate_plate(self, text):
        # Create a white plate with blue side strips
        img = Image.new('RGB', (self.width, self.height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Blue strips (approx 10% width each)
        strip_w = int(self.width * 0.08)
        draw.rectangle([0, 0, strip_w, self.height], fill=(0, 51, 153)) # Left strip
        draw.rectangle([self.width - strip_w, 0, self.width, self.height], fill=(0, 51, 153)) # Right strip
        
        # Draw text
        w, h = draw.textsize(text, font=self.font)
        draw.text(((self.width-w)/2, (self.height-h)/2 - 5), text, fill=(0, 0, 0), font=self.font)
        
        return np.array(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--out', type=str, default='datasets/synth_french_plates')
    parser.add_argument('--font', type=str, default='datasets/fonts/CharlesWright.ttf')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, 'hr'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'lr'), exist_ok=True)
    
    gen = FrenchPlateGenerator(args.font)
    data = []

    print(f"Generating {args.n} plates...")
    for i in tqdm(range(args.n)):
        text = gen.generate_text()
        plate_hr = gen.generate_plate(text)
        
        # Sample target width from dev-set distribution (28-138, mean 58)
        target_w = int(np.random.normal(58, 20))
        target_w = max(28, min(138, target_w))
        
        plate_lr, plate_sr_in = degrade_image(plate_hr, target_w)
        
        hr_path = f"hr/{i:06d}.png"
        lr_path = f"lr/{i:06d}.png"
        
        cv2.imwrite(os.path.join(args.out, hr_path), cv2.cvtColor(plate_hr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, lr_path), cv2.cvtColor(plate_lr, cv2.COLOR_RGB2BGR))
        
        data.append({'hr_path': hr_path, 'lr_path': lr_path, 'text': text})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.out, 'train.csv'), index=False)
    print("Generation complete.")

if __name__ == "__main__":
    main()
