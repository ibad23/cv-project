"""
Lightweight synthetic French license plate generator.

Stripped down from `plan.md` Workstream-1 to fit a 3-hour crash schedule:
  - 5k plates, not 50k
  - SIV (7-char LLDDDLL) + a few old-format 8-char plates
  - Single-stage degradation, not Real-ESRGAN second-order
  - Default font (DejaVu / Liberation), not Charles-Wright

Output layout:
  datasets/synth_french_plates/
    images/00000.png ... 04999.png
    train.csv  (filename,text)
"""
from __future__ import annotations
import argparse
import io
import os
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import pandas as pd

LETTERS = string.ascii_uppercase
DIGITS = string.digits

SIV_PAT = 'LLDDDLL'                 # 7 chars
OLD_PATS = ['DDDDLLDD', 'DDDLLLDD', 'DDLLLDDD', 'DDDLLDDD']  # 8 chars

def sample_text(rng: random.Random) -> str:
    if rng.random() < 0.85:
        pat = SIV_PAT
    else:
        pat = rng.choice(OLD_PATS)
    out = []
    for c in pat:
        if c == 'L':
            out.append(rng.choice(LETTERS))
        else:
            out.append(rng.choice(DIGITS))
    return ''.join(out)


# Try a few common bold fonts; fall back to default bitmap if none found.
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/mnt/c/Windows/Fonts/arialbd.ttf',
    '/mnt/c/Windows/Fonts/calibrib.ttf',
]

def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_plate_hr(text: str, rng: random.Random, w: int = 240, h: int = 50) -> Image.Image:
    """Render a clean, high-res French SIV-style plate. White background, black text."""
    img = Image.new('RGB', (w, h), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # SIV blue side bands (just visual flavor; OCR ignores them)
    band_w = int(w * 0.07)
    draw.rectangle([0, 0, band_w, h], fill=(0, 38, 100))
    draw.rectangle([w - band_w, 0, w, h], fill=(0, 38, 100))

    font_size = int(h * 0.78)
    font = load_font(font_size)

    # Center the text in the middle band
    inner_w = w - 2 * band_w
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = band_w + (inner_w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    # Slight jitter
    x += rng.randint(-2, 2)
    y += rng.randint(-1, 1)
    draw.text((x, y), text, fill=(15, 15, 15), font=font)

    # Optional dash for old-format 8-char plates
    if len(text) == 8 and rng.random() < 0.3:
        # SIV ones don't typically use dashes in labels but render is loose
        pass
    return img


def degrade(hr: Image.Image, rng: random.Random, target_input=(128, 32)) -> Image.Image:
    """Apply single-stage degradation matching dev-set distribution."""
    arr = np.array(hr)

    # 1. Bicubic downsample to a low-res width sampled to match dev distribution (28-138 px wide).
    target_w = rng.randint(28, 138)
    target_h = max(8, int(arr.shape[0] * target_w / arr.shape[1]))
    lr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 2. Random Gaussian blur
    sigma = rng.uniform(0.3, 1.5)
    k = max(3, int(2 * round(sigma * 2) + 1))
    if k % 2 == 0:
        k += 1
    lr = cv2.GaussianBlur(lr, (k, k), sigma)

    # 3. Random motion blur (sometimes)
    if rng.random() < 0.5:
        ksize = rng.choice([3, 5, 7])
        kernel = np.zeros((ksize, ksize))
        if rng.random() < 0.5:
            kernel[ksize // 2, :] = 1.0 / ksize     # horizontal
        else:
            kernel[:, ksize // 2] = 1.0 / ksize     # vertical
        lr = cv2.filter2D(lr, -1, kernel)

    # 4. JPEG re-encode
    q = rng.randint(40, 90)
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        lr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 5. Gaussian noise
    sigma_n = rng.uniform(2, 12)
    noise = rng.gauss(0, 1)  # placeholder to seed numpy via rng
    noise_arr = np.random.normal(0, sigma_n, lr.shape).astype(np.float32)
    lr = np.clip(lr.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)

    # 6. Resize to fixed model input
    out = cv2.resize(lr, target_input, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='datasets/synth_french_plates')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rows = []
    for i in range(args.n):
        text = sample_text(rng)
        hr = render_plate_hr(text, rng)
        lr = degrade(hr, rng)
        fname = f'{i:05d}.png'
        lr.save(img_dir / fname)
        rows.append({'filename': fname, 'text': text})
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{args.n} generated')

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'train.csv', index=False)
    print(f'Wrote {len(df)} samples to {out_dir}')

    # Spot-check: save a small grid for visual sanity
    sample_idx = rng.sample(range(args.n), min(16, args.n))
    tiles = []
    for idx in sample_idx:
        im = np.array(Image.open(img_dir / f'{idx:05d}.png').resize((128, 32)))
        tiles.append(im)
    grid = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, 16, 4)], axis=0)
    Image.fromarray(grid).save(out_dir / 'sample_grid.png')
    print(f'Wrote sample grid: {out_dir / "sample_grid.png"}')


if __name__ == '__main__':
    main()
