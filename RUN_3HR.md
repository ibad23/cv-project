# 3-Hour Crash Fine-Tune — Runbook

Hand-off doc for running the stripped-down fine-tune from your GPU machine.
Plan reference: `/home/ibad/.claude/plans/so-read-up-on-quirky-torvalds.md`.

## What's already done (in this repo)

- `code/synth/gen_french_plates_lite.py` — 5k-plate synthetic generator (CPU).
- `code/finetune/train_small_ocr.py` — CRNN-style trainer (ResNet/CRNN hybrid, 32×256 input, 8 per-position heads).
- `code/finetune/smallnet_engine.py` — inference wrapper with the `OCREngine` API.
- `datasets/synth_french_plates/` — **5,000 plates already generated** (CPU run, ~9 min). Sample grid at `datasets/synth_french_plates/sample_grid.png` — verified visually plausible.
- `code/ultraplate_pipeline.ipynb` — patched: `SmallNetOCREngine` is wired into `ENABLE`, `ENGINE_SR`, and `_OCR_BUILDERS`. Re-running the notebook will pick it up automatically once weights exist.

## What you need to run on the GPU box

### 1. Train the small OCR (~10–20 min on a modern GPU)

```bash
cd /path/to/cv-project
PYTHONUNBUFFERED=1 python3 code/finetune/train_small_ocr.py \
  --epochs 25 \
  --batch 128 \
  --lr 1e-3 \
  --workers 4 \
  --out models/small_ocr_french.pt
```

**What to watch for:**
- `device=cuda` on the first line. If it says `cpu`, the env is wrong — fix before continuing.
- `val_char_acc` should pass **0.6 by epoch 5** and **0.85 by epoch 12** on a working setup.
- If val_char_acc plateaus below 0.5 after 10 epochs, the architecture didn't converge — try `--lr 5e-4` or more epochs. The CPU sanity run on this repo only reached 0.29 because it was training a buggier ResNet-18 variant; the current CRNN-style net trains much faster.
- Best checkpoint auto-saves to `models/small_ocr_french.pt`. History JSON next to it.

### 2. Run the integrated pipeline

In Jupyter (or via `jupyter nbconvert --execute`):

```bash
jupyter nbconvert --to notebook --execute code/ultraplate_pipeline.ipynb \
  --output ultraplate_pipeline_executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Or open `code/ultraplate_pipeline.ipynb` and run all cells. The new engine `smallnet` will appear in `Loaded OCR engines:` after cell 21.

### 3. Capture results for the slide

The notebook already produces a per-sequence breakdown table and a submission CSV. Compare:
- Score with `ENABLE["smallnet"] = False` (baseline) vs. `True` (with fine-tune).
- Re-run the eval cells (section 8) for both. Note: lazy-loaded engines are cached, so toggle and re-run; or just record both runs end-to-end.

## Fallback if training fails or smallnet hurts performance

In `code/ultraplate_pipeline.ipynb` cell 2, set:
```python
"smallnet": False
```
and re-run. The pipeline reverts to the unmodified ultraplate baseline.

## Files modified in this repo (commit-ready)

- `code/ultraplate_pipeline.ipynb` — added smallnet engine + registration (3 small edits).
- `code/finetune/train_small_ocr.py` — NEW.
- `code/finetune/smallnet_engine.py` — NEW.
- `code/synth/gen_french_plates_lite.py` — NEW.
- `code/ultraplate_v2_pipeline.ipynb` — earlier scoring assert fix (unrelated to fine-tune).
- `datasets/synth_french_plates/` — 5k generated plates + train.csv.
- `models/small_ocr_french.pt` — present from CPU sanity run; **overwrite by GPU training before evaluating**.
