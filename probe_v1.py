"""Inference-time probes on a trained v1 checkpoint (hook in model.py; see loupe_v2.probe_transform).
Analysis only, never selection. Use the test set: a v1 checkpoint was trained on all of train, so the
carved val split is not held out for it, and its 'none' accuracy was picked on test (read the deltas).

    python probe_v1.py --weights output/best_model.pth --out results/probe_v1_swin_b_test.json   # Swin-B
    python probe_v1.py --weights ... --model swin_tiny_patch4_window7_224
"""
import argparse
import hashlib
import json
import os

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from tqdm import tqdm

from model import SwinWithLoupe

MODES = ["none", "no_erasure", "no_contrast", "gate_off", "floor"]


@torch.no_grad()
def run(model, loader, device, mode):
    model.probe_mode = mode
    correct, n, stats = 0, 0, {}
    for images, labels in tqdm(loader, desc=mode, leave=False):
        logits, m = model(images.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        n += labels.size(0)
        if mode == "none":  # how the trained v1 mask sits relative to the LN-invariance argument
            gmax, gmin = F.max_pool2d(m, 2), -F.max_pool2d(-m, 2)
            for k, v in {"coverage": m.mean(), "frac_m_lt_1e-3": (m < 1e-3).float().mean(),
                         "frac_groups_erased": (gmax < 1e-3).float().mean(),
                         "median_group_contrast": (gmax / gmin.clamp_min(1e-6)).median()}.items():
                stats[k] = stats.get(k, 0.0) + v.item() / len(loader)
    return correct / n, stats


def probe(weights, model_name="swin_base_patch4_window7_224", data_dir="./data", split="test", modes=MODES,
          split_file="splits/cub_val_frac0.1_seed0.json", out=None, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwinWithLoupe(model_name=model_name, num_classes=200, pretrained=False)
    missing, unexpected = model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    assert not missing, f"checkpoint lacks {missing[:5]}"
    assert all(k.endswith(("relative_position_index", "attn_mask")) for k in unexpected), unexpected[:5]
    model.to(device).eval()

    tf = create_transform(input_size=224, is_training=False)
    if split == "test":
        ds = ImageFolder(os.path.join(data_dir, "test"), transform=tf)
    else:
        root = os.path.join(data_dir, "train")
        ds = ImageFolder(root, transform=tf)
        with open(split_file) as f:
            keep = set(json.load(f)["val"])
        ds = Subset(ds, [i for i, (p, _) in enumerate(ds.samples) if os.path.relpath(p, root) in keep])
    if limit:
        ds = Subset(ds, range(limit))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    with open(weights, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]
    res = {"weights": weights, "weights_sha": sha, "model": model_name, "split": split, "n_images": len(ds),
           "timm": timm.__version__, "modes": {}}
    for mode in modes:
        acc, stats = run(model, loader, device, mode)
        res["modes"][mode] = {"acc": acc, **stats}
        print(json.dumps({"mode": mode, "acc": round(acc, 4), **{k: round(v, 4) for k, v in stats.items()}}))
    model.probe_mode = "none"
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="swin_base_patch4_window7_224", help="v1 default (paper model)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--split_file", default="splits/cub_val_frac0.1_seed0.json", help="used with --split val")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--limit", type=int, help="smoke test: cap images")
    args = ap.parse_args()
    probe(args.weights, args.model, args.data_dir, args.split, args.modes, args.split_file, args.out, args.limit)


if __name__ == "__main__":
    main()
