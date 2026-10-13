# Loupe v2 experimental protocol

Frozen 2026-10-13. Applies to every v2 run, on every dataset, with the baseline and every Loupe
row trained in the same harness (`train_v2.py`). Changing anything below starts a new protocol
version, and results from different versions are not compared.

## Backbones and software

| | timm tag | Params (200-way head) | v1 Loupe params |
|---|---|---|---|
| Swin-T (default) | `swin_tiny_patch4_window7_224.ms_in1k` | 27,673,154 | 83,041 (0.300%) |
| Swin-B (final rows only) | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 86,948,224 | 147,585 (0.170%) |

- **Software:** timm 1.0.30, pinned in each config and asserted at start-up. The torch version is recorded per run.
- **Classifier:** `reset_classifier(num_classes)`, with default `nn.Linear` init.
- **Overhead:** added parameters divided by backbone parameters, including the new classifier.
- **Swin-T tag:** frozen as `ms_in1k` because that is what the v1 code ran. Timm also has `ms_in22k_ft_in1k` for Swin-T, which would match Swin-B's pretraining. Switch only before the Phase 1 screen, never mid-study.

## Optimisation

- **Optimizer:** Lion (`timm.optim.create_optimizer_v2`), lr 1e-5, betas (0.9, 0.99), **weight decay 0**, one parameter group. The only exception is a learned α, which gets its own lr of 1e-3 (Phase 2).
- **Schedule:** cosine annealing stepped per epoch, T_max 50, eta_min 1e-7, no warm-up.
- **Batch:** Swin-T uses batch 16 × gradient accumulation 2 (32 effective) in fp32. Swin-B uses 8 × 4 (32 effective) with fp16 AMP, and activation checkpointing only if it runs out of memory. AMP is not implemented yet; it must be added before any Swin-B run, and the Swin-B baseline uses the same settings.
- **Loss:** cross-entropy plus the row's regulariser. The baseline uses cross-entropy only, with no label smoothing.

## Augmentation

- **Train:** `create_transform(224, is_training=True, auto_augment='rand-m9-mstd0.5')`. That is:
  - RandomResizedCrop(224, scale 0.08–1, bilinear)
  - horizontal flip, p = 0.5
  - RandAugment(n = 2, m = 9, mstd = 0.5)
  - ImageNet mean/std normalisation
  - no colour jitter, random erasing, mixup/cutmix or label smoothing
- **Eval:** resize the shorter side to 256 (bilinear, antialiased), then centre-crop 224.

## Epochs, early stopping and selection

- **Epochs:** at most 50.
- **Per-epoch validation:** loss (cross-entropy plus the regulariser, as in v1), accuracy, and `gate_stats()`.
- **Early stopping:** stop when validation loss has not improved by more than 1e-3 for 5 epochs.
- **Checkpoint:** the epoch with the highest validation accuracy.
- **Selection:** every choice (row, λ/τ, α, placement, add-ons) is made on validation accuracy.
- **Test:** evaluated once per final checkpoint, never used for selection.
  - Screens run with `eval_test: false` and never touch test.
  - Paper runs set `eval_test: true`.
  - Opening test later for a finished run evaluates its saved checkpoint once and does not retrain.

## Splits

- **CUB-200-2011:** official train/test split (5994 / 5794).
  - Validation is 10% per class of the official train set, stratified with seed 0: 600 images (3 per class), leaving 5394 for training.
  - The split lives in `splits/cub_val_frac0.1_seed0.json`, which is committed, and the harness refuses to overwrite it.
  - Every row and every seed uses it, and the harness checks it against the files on disk.
- **Other datasets:** each gets exactly one split file, made the same way (`train_v2.py --make_split`) before its first run.

## Seeds

- **Screening:** seed 0, one run.
- **Paper numbers:** seeds 0, 1 and 2, reported as mean ± sample std (n − 1).
- **What the seed controls:** classifier and mask-head init, data order, augmentation, and DropPath. The split seed stays 0 regardless.

## Recorded for every run

These go in `config.json`, `log.jsonl` and `results.json`:

- exact pretrained tag and HF hub id
- timm, torch and Python versions
- GPU
- backbone and added parameters, and overhead %
- sha256 of `loupe_v2.py` + `train_v2.py`
- per-epoch log: train/validation loss, validation accuracy and `gate_stats()`
- wall-clock GPU-hours

## Where v1 / arXiv v2 differ from this protocol

| Item | arXiv v2 says | v1 code did |
|---|---|---|
| Swin-T pretraining | ImageNet-21k | `ms_in1k` |
| Weight decay | 0.02 | 0 |
| Augmentation | crop + flip + colour jitter | RandAugment, no jitter |
| Batch | 32 | 16, no accumulation |
| Overhead | < 0.1% | 0.17% (B), 0.30% (T) |
| Validation and early stopping | a validation split; early stopping on accuracy | the test set: early stopping on test loss, checkpoint chosen by highest test accuracy |
| Seeds | 3 | none set |

The CUB authors warn that CUB images overlap ImageNet. Every row shares the same backbone, so
comparisons stay controlled, but absolute accuracies inherit that overlap.
