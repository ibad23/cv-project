"""
Inference wrapper for the small fine-tuned OCR model.

Designed to slot into ultraplate_pipeline.ipynb's OCREngine API:
    engine = SmallNetOCREngine(weights_path)
    text, per_char_conf = engine.read(img_bgr)

The model is a ResNet-18 backbone -> AdaptiveAvgPool2d((1, 8)) -> shared Linear(512, 37)
producing per-position softmax over the 37-symbol alphabet (A-Z, 0-9, '_' blank).
Trained with 32x256 input.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

NUM_CLASSES = 37
NUM_POS = 8
INPUT_HW = (32, 256)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class _SmallOCR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_pos=NUM_POS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, (2, 3), padding=(0, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, num_pos))
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = self.pool(f)
        f = f.squeeze(2).transpose(1, 2)
        return self.head(f)


class SmallNetOCREngine:
    """Mimics the OCREngine API used in ultraplate_pipeline.ipynb."""
    name = 'smallnet'
    preferred_sr = 'bicubic'   # synthetic data was bicubic-degraded

    def __init__(self, weights_path: str | Path = 'models/small_ocr_french.pt',
                 device: str | None = None):
        ckpt = torch.load(weights_path, map_location='cpu')
        self.alphabet = ckpt.get('alphabet', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'))
        self.num_pos = ckpt.get('num_pos', NUM_POS)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = _SmallOCR(num_classes=len(self.alphabet), num_pos=self.num_pos).to(self.device)
        state = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize(INPUT_HW),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ])

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def read(self, img: np.ndarray):
        """Return (text_alphanum_only, per_char_conf_list) — matches OCREngine API."""
        x = self._preprocess(img)
        logits = self.model(x)              # [1, P, C]
        probs = logits.softmax(-1)[0]        # [P, C]
        idxs = probs.argmax(-1).cpu().numpy()
        confs = probs.max(-1).values.cpu().numpy()

        chars, out_confs = [], []
        blank_idx = self.alphabet.index('_')
        for i, c in zip(idxs, confs):
            if int(i) == blank_idx:
                continue
            ch = self.alphabet[int(i)]
            if ch.isalnum():
                chars.append(ch.upper())
                out_confs.append(float(c))
        return ''.join(chars), out_confs

    @torch.no_grad()
    def read_dists(self, img: np.ndarray):
        """Return list of {char: prob} dicts per position (length NUM_POS).

        Useful for grammar-aware downstream voting; not used by ultraplate_pipeline
        directly but kept for the v2 path.
        """
        x = self._preprocess(img)
        probs = self.model(x).softmax(-1)[0].cpu().numpy()  # [P, C]
        out = []
        for p in probs:
            out.append({c: float(p[i]) for i, c in enumerate(self.alphabet)})
        return out
