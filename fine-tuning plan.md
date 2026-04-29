# Plan — XLPSR Fine-Tuning Strategy (handoff for implementer)

## Context

The team is at the XLPSR ICIP 2026 challenge. Current dev-set scores:

| Notebook | Dev score (564 max) | Exact matches |
|---|---|---|
| `demo.ipynb` (Swin2SR + PaddleOCR) | −72 (−12.8%) | 0/39 |
| `sota_pipeline.ipynb` | −4 (−0.7%) | 0/39 |
| `ultraplate_pipeline.ipynb` | −22 (−3.9%) | 0/39 |

`ultraplate` has the best fusion stack (ECC sub-pixel registration + Needleman–Wunsch + per-engine SR routing) but **scores worse than `sota`** because its OCR engines are zero-shot and biased toward English/Latin scene text. **Both pipelines bottleneck on the same thing: zero-shot OCR engines that hallucinate dictionary words on small French plates and an SR network that optimises pixels instead of legibility.**

The fix supported by the latest literature (Nascimento 2024 — LCDNet; Robust ALPR via Synthetic Data 2025) is to fine-tune the two stages with plate-specific objectives. We have ~24 hr on a T4/V100, both SR+OCR in scope, and synthetic French SIV + the qanastek FrenchLicencePlateDataset as training data.

Goal: turn the existing `ultraplate_pipeline.ipynb` from −3.9 % into positive territory with first-exact-match plates by swapping in two fine-tuned weight files.

## Recommendation in one paragraph

Build a **shared synthetic French-plate generator** once, then run **two fine-tunes in parallel** on it: (1) `fast-plate-ocr` MobileViTV2-European → French-SIV-specialised, (2) Real-ESRGAN x4 → LCDNet-style OCR-aware fine-tune using the *frozen* fine-tuned fast-plate-ocr as perceptual discriminator. Drop both weight files into `ultraplate_pipeline.ipynb` as path overrides. Keep the fusion / Needleman–Wunsch / format-constraint code untouched — it is the strongest part of the stack.

We do **not** add diffusion SR (CharDiff-LP / DiffPlate), TrOCR fine-tune, or Qwen-VL adjudication in this round — they are all too expensive for a single T4, the OCR fine-tune already kills the dictionary-bias failure mode they were proposed to fix, and they can be a Round-2 add-on if the gain is below target.

## Workstream 1 — Synthetic French SIV plate generator

**Owner:** implementer, day 0 (a few hours, CPU only).
**Output:** `code/synth/gen_french_plates.py` and a `synth_french_plates/` folder with 50k images + a `train.csv` manifest of `(image_path, text)` rows.

Generation recipe (standard 2025 ALPR practice):

1. Render plate text on the official French SIV background using the **Marianne** / **Plaque-Immatriculation** font (the open-source `Charles Wright 2001` clone is fine if the official is unavailable).
2. Sample text uniformly from the two French format families documented in `memory/project_xlpsr.md`:
   - SIV (post-2009): `AA-NNN-AA` → 7 chars `[A-Z]{2}[0-9]{3}[A-Z]{2}`.
   - Old: `NNNN-AA-NN`, `NNN-AAA-NN`, `NN-AAA-NNN`, `NNN-AA-NNN` (cover the 1-3 trailing digit variants).
3. Apply Real-ESRGAN's **second-order degradation pipeline** to produce a paired LR (used for SR fine-tune) and a "hard-readable" version (used for OCR fine-tune): blur → noise → JPEG → bicubic-down → blur → noise → JPEG → bicubic-up. Plate widths sampled from the dev-set distribution **28-138 px (mean 58)** so the synthetic distribution matches the test distribution.
4. Add the 211 qanastek/FrenchLicencePlateDataset images as a held-out *real-world* validation split — never seen in training.

Reuse: there is no existing generator in the repo; build from scratch. Existing degradation utilities can be cribbed from Real-ESRGAN's `realesrgan/data/realesrgan_dataset.py` upstream (BSD-licensed, copy directly).

Critical: do **NOT** train on the dev set's 39 sequences — it is the eval set and any leak voids the score we report internally.

## Workstream 2 — Fine-tune `fast-plate-ocr` for French SIV

**Owner:** implementer, day 1 (~3-6 hr T4).
**Base model:** `cct-s-v2-global-model` (the European MobileViTV2 already covers French alphabet — we just need to specialise the head + last 2 transformer blocks).
**Why this engine, not TrOCR:** per-character softmax over `[0-9A-Z_]` only — *cannot* hallucinate "TOTAL"/"NOSE"/"JNAY", which is the dominant `−1`-penalty failure mode logged in `memory/project_xlpsr.md`.

Steps:

1. `pip install fast-plate-ocr[train]` and follow `examples/fine_tune_workflow.ipynb` from the upstream repo as the template.
2. Training config:
   - Optimizer Adam, lr `5e-5`, batch 64, 10 epochs, cross-entropy on per-position softmax.
   - Freeze MobileViTV2 stem; unfreeze last 2 transformer blocks + head.
   - Augmentation on the fly: ±5° rotation, ±10% gamma, motion blur (matches `ultraplate` TTA).
   - 90 / 10 train-val split on the 50k synthetic; the 211 qanastek plates are an **additional** held-out test for sanity-checking.
3. Export weights as `models/fastplate_french_siv.onnx` (and `.pt` for the python loader).
4. Smoke-test by running on the 8 dev-set sequences `ultraplate_pipeline.ipynb` already evaluates and confirming per-character accuracy beats the upstream model on at least 6 of 8.

Expected gain (per `ultraplate_pipeline.ipynb` §13A): **+20-30 score points**, first-exact-match plates appear.

## Workstream 3 — Fine-tune Real-ESRGAN with LCDNet-style OCR-aware loss

**Owner:** implementer, day 2 overnight (~12-18 hr T4).
**Base model:** Real-ESRGAN x4 RRDBNet (already wired in `ultraplate_pipeline.ipynb`).
**Reference paper:** Nascimento et al. 2024, "Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach" (arxiv 2408.15103). **Code is public** — clone the LCDNet repo and use it as the training scaffold; we do not need to reimplement.

Loss: `L = λ1 * L1(SR, HR) + λ2 * L_perceptual(VGG) + λ3 * L_OCR`, where `L_OCR` is the CTC / per-position cross-entropy of the **frozen fine-tuned `fast-plate-ocr` from Workstream 2** evaluated on the SR output against the GT text. λ3 ≈ 0.1 to start (LCDNet defaults). This is exactly the LCOFL recipe but with our OCR head as the discriminator.

Training data: pairs from Workstream 1 — HR is the clean rendered plate, LR is the degraded version. 50 epochs, lr `1e-4` cosine, batch 16 at 64×128 px.

Output: `models/realesrgan_x4_french_lcd.pth` — drop-in replacement for the existing Real-ESRGAN weight path in `ultraplate_pipeline.ipynb` §3.

Expected gain (LCDNet paper UFPR-SR-Plates): **31% → 44.7% recognition** at the SR stage; in our pipeline this compounds with the OCR fine-tune.

## Workstream 4 — Integration & evaluation

**Owner:** implementer, day 3 (~2 hr CPU).

1. In `code/ultraplate_pipeline.ipynb`:
   - Override the fast-plate-ocr model path → `models/fastplate_french_siv.onnx`.
   - Override the Real-ESRGAN weight path → `models/realesrgan_x4_french_lcd.pth`.
   - Leave §2 (ECC + median fusion), §6 (Needleman–Wunsch), and §5 (TTA) **unchanged** — those are the strongest parts.
2. Re-run the full 39-sequence dev-set eval cell; record total score, exact-match count, and per-sequence delta vs. the pre-fine-tune ultraplate run logged in the project memory.
3. Per-engine ablation: also evaluate `fastplate_french` alone (no PARSeq / Paddle) — based on ultraplate's findings the other engines may now be dragging the ensemble down.

**Success criterion for Round 1:** total dev score > 0, exact-match count ≥ 3/39. If only one of the two is hit, ship the OCR fine-tune alone (it is the higher-confidence win).

## Critical files

- `code/ultraplate_pipeline.ipynb` — integration target (only model paths edited).
- `code/sota_pipeline.ipynb` — keep as a fallback baseline; do not modify.
- `code/synth/gen_french_plates.py` — **new**, Workstream 1 output.
- `code/finetune/train_fastplate.py` — **new**, Workstream 2 training script.
- `code/finetune/train_realesrgan_lcd.py` — **new**, forked from LCDNet repo.
- `models/fastplate_french_siv.onnx` — **new**, Workstream 2 output.
- `models/realesrgan_x4_french_lcd.pth` — **new**, Workstream 3 output.
- `challenge_development_set_final/` — eval set, **do not** train on this.

## Verification

End-to-end check the implementer must run before declaring done:

1. `python code/synth/gen_french_plates.py --n 100 --out /tmp/sanity` and visually inspect 5 outputs — text rendered correctly, degradation looks like real LR plates.
2. After Workstream 2, run the upstream `fast_plate_ocr` smoke test on 10 qanastek images — character accuracy must beat the unmodified European model.
3. After Workstream 3, eyeball 5 SR outputs vs. the un-fine-tuned Real-ESRGAN on dev sequences 001 / 017 (both have logged baseline numbers in the project memory).
4. Run the full dev-set eval cell in `ultraplate_pipeline.ipynb` and confirm score > 0 and ≥ 1 exact match. Save the eval cell output back into the notebook so it shows up in the team's git history.

## Out of scope for this round (keep on the radar)

- Fine-tune TrOCR-Base on the same synthetic data — only worth adding if dev score plateaus and we still have the gated-ensemble slot in `ultraplate` §13C unfilled.
- Diffusion-SR (CharDiff-LP, DiffPlate) — better PSNR in papers but too slow for the inference budget and the laptop constraint.
- Qwen2.5-VL-3B / InternVL3 final-pass adjudicator — zero-shot, not a fine-tune; revisit only if Round 1 still has < 5 exact matches.
- UFPR-SR-Plates pre-train for the SR model — Brazilian plates differ in font/aspect; user excluded it from the data list. Can revisit as an optional pre-train if Round 1 falls short.
