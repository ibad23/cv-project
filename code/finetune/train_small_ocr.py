"""
Crash-train a small per-position OCR model for French license plates.

Architecture:
  - timm ResNet-18 (pretrained), no head, no global pool: feature map [B, 512, H/32, W/32]
  - AdaptiveAvgPool2d to (1, 8): [B, 512, 1, 8] -> [B, 8, 512]
  - Shared Linear(512, 37) head applied across the 8 position tokens.
  - 37-way alphabet: A-Z (26) + 0-9 (10) + '_' blank (1).

Training:
  - AdamW lr=1e-4 wd=1e-4, cosine schedule, batch 64.
  - Per-position cross-entropy with label_smoothing=0.05.
  - 8 epochs, save best by per-character val accuracy.
  - 90/10 train/val split.

Output:
  models/small_ocr_french.pt   (state_dict + alphabet metadata)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')  # 37 classes; '_' is blank/pad
CHAR2IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX2CHAR = {i: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET)
NUM_POS = 8


def encode(text: str) -> torch.Tensor:
    text = text.upper().ljust(NUM_POS, '_')[:NUM_POS]
    return torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)


def decode(idxs) -> str:
    return ''.join(IDX2CHAR[int(i)] for i in idxs)


class PlateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        # Light augmentation (CPU-side, on PIL).
        if train:
            self.tf = T.Compose([
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), fill=128),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.Resize((32, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((32, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row['filename']).convert('RGB')
        x = self.tf(img)
        y = encode(str(row['text']))
        return x, y


class SmallOCR(nn.Module):
    """Shallow CRNN-style CNN. Input 32x256 -> [B,512,1,32] -> pool to [B,512,1,8]
    -> [B, 8, 512] -> Linear(512, num_classes). Each output position covers
    32 input pixels (= one character width) with a receptive field of ~50 px,
    so per-position classification is well-matched.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, num_pos: int = NUM_POS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                   # 16x128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                   # 8x64
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                                              # 4x64
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                                              # 2x64
            nn.Conv2d(512, 512, (2, 3), padding=(0, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),                        # 1x64
        )
        self.pool = nn.AdaptiveAvgPool2d((1, num_pos))
        self.head = nn.Linear(512, num_classes)
        self.num_pos = num_pos

    def forward(self, x):
        f = self.cnn(x)                    # [B, 512, 1, 64]
        f = self.pool(f)                   # [B, 512, 1, num_pos]
        f = f.squeeze(2).transpose(1, 2)   # [B, num_pos, 512]
        return self.head(f)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    exact = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)             # [B, P, C]
            preds = logits.argmax(-1)     # [B, P]
            correct += (preds == y).sum().item()
            total += y.numel()
            exact += (preds == y).all(dim=1).sum().item()
    return correct / max(total, 1), exact / max(len(loader.dataset), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='datasets/synth_french_plates')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=str, default='models/small_ocr_french.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data)
    df = pd.read_csv(data_dir / 'train.csv')
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(50, int(0.1 * len(df)))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]
    print(f'train={len(train_df)}  val={len(val_df)}')

    train_ds = PlateDataset(train_df, data_dir / 'images', train=True)
    val_ds = PlateDataset(val_df, data_dir / 'images', train=False)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')
    model = SmallOCR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)                                        # [B, P, C]
            loss = crit(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)
        char_acc, exact_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f'epoch {epoch+1:2d}/{args.epochs}  loss={train_loss:.4f}  '
              f'val_char_acc={char_acc:.4f}  val_exact={exact_acc:.4f}  ({dt:.0f}s)')
        history.append({'epoch': epoch + 1, 'loss': train_loss,
                        'val_char_acc': char_acc, 'val_exact': exact_acc, 'time_s': dt})
        if char_acc > best_acc:
            best_acc = char_acc
            torch.save({
                'state_dict': model.state_dict(),
                'alphabet': ALPHABET,
                'num_pos': NUM_POS,
                'val_char_acc': char_acc,
                'val_exact': exact_acc,
                'epoch': epoch + 1,
            }, out_path)
            print(f'  -> saved checkpoint to {out_path} (best so far)')

    print(f'\nDone. best val_char_acc={best_acc:.4f}  total={time.time()-t_start:.0f}s')
    with open(out_path.with_suffix('.history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
