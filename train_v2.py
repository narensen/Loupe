"""Harness for every Loupe v2 experiment (recipe frozen in PROTOCOL.md). Same recipe as v1 train.py
(Lion, cosine per epoch, RandAugment, early stopping on val loss incl. the reg term), but validation is
carved from train, the checkpoint is chosen on validation accuracy, and test is evaluated at most once
per final checkpoint (only when the config sets eval_test). Runs resume from <out>/<run>/last.pth, so a
Colab disconnect costs at most one epoch.

    python train_v2.py --config configs/screen_cub_swin_t.json --row A
    python train_v2.py --config configs/screen_cub_swin_t.json --row D --gate_pos postnorm
    python train_v2.py --config configs/screen_cub_swin_t.json --summarize results/screen_cub_swin_t
    python train_v2.py --config configs/screen_cub_swin_t.json --make_split   # once per dataset, then commit
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import random
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from timm.optim import create_optimizer_v2
from tqdm import tqdm

from loupe_v2 import build_loupe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


class Plain(nn.Module):
    """No-Loupe baseline with the LoupeSwin training interface: same build, harness and recipe."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def reg_loss(self, **kw):
        return 0.0

    def gate_stats(self):
        return {}


def build_model(cfg, row, pretrained=True):
    row = dict(row)
    if row.pop("loupe", True):
        return build_loupe(cfg["model"], cfg["num_classes"], pretrained=pretrained, **row)
    bb = timm.create_model(cfg["model"], pretrained=pretrained)  # same RNG order as build_loupe
    bb.reset_classifier(cfg["num_classes"])
    return Plain(bb)


# ---- data -------------------------------------------------------------------------------------
def make_split(cfg):
    """Stratified per-class val split of the official train set. Written once per dataset, then committed."""
    path, root = cfg["split_file"], os.path.join(cfg["data_dir"], "train")
    assert not os.path.exists(path), f"{path} exists; splits are frozen (PROTOCOL.md)"
    ds = ImageFolder(root)
    by_cls = {}
    for i, (_, y) in enumerate(ds.samples):
        by_cls.setdefault(y, []).append(i)
    rng = random.Random(cfg["split_seed"])
    val = sorted(i for idx in by_cls.values() for i in rng.sample(idx, max(1, round(cfg["val_frac"] * len(idx)))))
    split = {"frac": cfg["val_frac"], "seed": cfg["split_seed"], "n_train_total": len(ds.samples),
             "n_classes": len(ds.classes), "val": [os.path.relpath(ds.samples[i][0], root) for i in val]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=0)
    return split


def load_split(cfg, samples, root):
    """Boolean val mask over `samples`; fails loudly if the data on disk does not match the split file."""
    with open(cfg["split_file"]) as f:
        split = json.load(f)
    rel = [os.path.relpath(p, root) for p, _ in samples]
    assert len(rel) == split["n_train_total"], f"{root} has {len(rel)} images, split expects {split['n_train_total']}"
    val = set(split["val"])
    is_val = [r in val for r in rel]
    assert sum(is_val) == len(val), "split file lists images that are not in the train folder"
    return is_val


def make_loaders(cfg, g, limit=None):
    root = os.path.join(cfg["data_dir"], "train")
    train_tf = create_transform(input_size=cfg["img_size"], is_training=True, auto_augment=cfg["auto_augment"])
    eval_tf = create_transform(input_size=cfg["img_size"], is_training=False)
    train_full, val_full = ImageFolder(root, transform=train_tf), ImageFolder(root, transform=eval_tf)
    is_val = load_split(cfg, train_full.samples, root)
    train_ds = Subset(train_full, [i for i, v in enumerate(is_val) if not v][:limit])
    val_ds = Subset(val_full, [i for i, v in enumerate(is_val) if v][:limit])
    test_ds = ImageFolder(os.path.join(cfg["data_dir"], "test"), transform=eval_tf)
    if limit:
        test_ds = Subset(test_ds, range(limit))
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, generator=g, **kw),
            DataLoader(val_ds, shuffle=False, **kw), DataLoader(test_ds, shuffle=False, **kw))


# ---- helpers ----------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, reg, desc):
    model.eval()
    loss, correct, n, stats = 0.0, 0, 0, {}
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss += (F.cross_entropy(logits, labels) + model.reg_loss(**reg)).item()
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
        for k, v in model.gate_stats().items():
            stats[k] = stats.get(k, 0.0) + v / len(loader)
    return loss / len(loader), correct / n, stats


def test_metrics(model, loader, reg):
    loss, acc, stats = evaluate(model, loader, reg, "test (once)")
    return {"test_acc": acc, "test_loss": loss, **{f"test_{k}": v for k, v in stats.items()}}


def run_meta(model):
    """What every run records: exact pretrained tag, versions, GPU, parameter overhead, code hash."""
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_add = sum(p.numel() for p in model.parameters()) - n_bb
    code = hashlib.sha256()
    for f in ("loupe_v2.py", "train_v2.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            code.update(fh.read())
    pc = model.backbone.pretrained_cfg
    return {"pretrained_tag": f"{pc['architecture']}.{pc['tag']}", "hf_hub_id": pc.get("hf_hub_id"),
            "timm": timm.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "backbone_params": n_bb, "added_params": n_add, "overhead_pct": 100 * n_add / n_bb,
            "code_sha": code.hexdigest()[:12]}


def save_atomic(obj, path):  # a disconnect mid-write must not corrupt the checkpoint on Drive
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def rng_state(g):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "loader": g.get_state()}


def set_rng_state(s, g):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    g.set_state(s["loader"])


def run_name(row_name, gate_pos, seed):
    return f"{row_name}{'_' + gate_pos if gate_pos else ''}_seed{seed}"


def load_result(out, row_name, seed, gate_pos=None):
    with open(os.path.join(out, run_name(row_name, gate_pos, seed), "results.json")) as f:
        return json.load(f)


# ---- training ---------------------------------------------------------------------------------
def train(config, row_name, gate_pos=None, seed=None, out=None, limit=None, epochs=None):
    """Train one row and return its results. Finished runs are skipped; interrupted runs resume."""
    with open(config) as f:
        cfg = json.load(f)
    cfg.update({k: v for k, v in (("seed", seed), ("epochs", epochs)) if v is not None})
    row = dict(cfg["rows"][row_name], **({"gate_pos": gate_pos} if gate_pos else {}))
    assert not row.get("loupe", True) or row.get("gate_pos"), \
        f"row {row_name}: pass gate_pos (the better of B / B2 on validation accuracy)"
    assert timm.__version__ == cfg["timm_version"], f"timm {timm.__version__} installed, config pins {cfg['timm_version']}"
    run = run_name(row_name, gate_pos, cfg["seed"])
    out_dir = os.path.join(out or os.path.join("output", os.path.splitext(os.path.basename(config))[0]), run)
    os.makedirs(out_dir, exist_ok=True)
    res_path, best_path, last_path, log_path = (os.path.join(out_dir, f)
                                                for f in ("results.json", "best.pth", "last.pth", "log.jsonl"))
    reg, sig = cfg["reg"], json.dumps({**cfg, "rows": None, "row": row}, sort_keys=True)

    if os.path.exists(res_path):  # finished; open test now only if the config asks and it was never opened
        with open(res_path) as f:
            res = json.load(f)
        if cfg["eval_test"] and "test_acc" not in res:
            t0 = time.time()
            model = build_model(cfg, row, pretrained=False).to(DEVICE)
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
            res.update(test_metrics(model, make_loaders(cfg, torch.Generator(), limit)[2], reg))
            res["gpu_hours"] += (time.time() - t0) / 3600
            with open(res_path, "w") as f:
                json.dump(res, f, indent=2)
        if os.path.exists(last_path):
            os.remove(last_path)
        print(f"[{run}] done: best val acc {100 * res['best_val_acc']:.2f}% at epoch {res['best_epoch']}"
              + (f", test {100 * res['test_acc']:.2f}%" if "test_acc" in res else ""))
        return res

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader, val_loader, test_loader = make_loaders(cfg, g, limit)
    torch.manual_seed(cfg["seed"])  # init independent of how much RNG the data setup consumed
    model = build_model(cfg, row).to(DEVICE)
    meta = run_meta(model)
    optimizer = create_optimizer_v2(model, opt=cfg["opt"], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["eta_min"])
    st = {"epoch": 0, "best_acc": -1.0, "best_epoch": -1, "best_val_loss": float("inf"), "patience": 0, "elapsed": 0.0}

    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        assert ck["sig"] == sig, f"config changed since {run} started; delete {out_dir} to restart it"
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"], g)
        st = ck["state"]
        if st["best_epoch"] == st["epoch"]:  # disconnect may have hit between last.pth and best.pth
            save_atomic(model.state_dict(), best_path)
        with open(log_path) as f:  # drop log lines from the epoch that was cut off
            keep = [l for l in f if json.loads(l)["epoch"] <= st["epoch"]]
        with open(log_path, "w") as f:
            f.writelines(keep)
        print(f"[{run}] resuming after epoch {st['epoch']}")
    else:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({**cfg, "rows": None, "row_name": row_name, "row": row, "meta": meta}, f, indent=2)
        open(log_path, "w").close()
        print(f"[{run}] {meta['pretrained_tag']} | +{meta['added_params']:,} params ({meta['overhead_pct']:.3f}%) | "
              f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val | {meta['gpu']}")

    accum, n_steps, t0 = cfg["accum_steps"], len(train_loader), time.time()
    for epoch in range(st["epoch"], cfg["epochs"]):
        if st["patience"] >= cfg["patience"]:
            break
        t_ep = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"{run} ep{epoch + 1} train", leave=False)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = F.cross_entropy(model(images), labels) + model.reg_loss(**reg)
            group = min(accum, n_steps - step // accum * accum)  # short final group at epoch end
            (loss / group).backward()
            if (step + 1) % accum == 0 or step + 1 == n_steps:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()

        val_loss, val_acc, stats = evaluate(model, val_loader, reg, f"{run} ep{epoch + 1} val")
        rec = {"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0], "train_loss": train_loss / n_steps,
               "val_loss": val_loss, "val_acc": val_acc, **{f"val_{k}": v for k, v in stats.items()}}
        scheduler.step()
        new_best = val_acc > st["best_acc"]
        if new_best:
            st["best_acc"], st["best_epoch"] = val_acc, epoch + 1
        if st["best_val_loss"] - val_loss > cfg["min_delta"]:
            st["best_val_loss"], st["patience"] = val_loss, 0
        else:
            st["patience"] += 1
        st["epoch"] = epoch + 1
        st["elapsed"] += time.time() - t0
        t0 = time.time()
        rec["epoch_seconds"] = round(t0 - t_ep, 1)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng": rng_state(g), "state": st, "sig": sig}, last_path)
        if new_best:
            save_atomic(model.state_dict(), best_path)
        gate = "".join(f" | {k} {stats[k]:.3g}" for k in ("coverage", "frac_groups_erased", "median_group_contrast") if k in stats)
        print(f"[{run}] ep {epoch + 1:2d} | train {rec['train_loss']:.3f} | val loss {val_loss:.3f} acc {100 * val_acc:.2f}%"
              f"{' *' if new_best else ''}{gate} | patience {st['patience']}/{cfg['patience']}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    with open(log_path) as f:
        best_rec = next(r for r in map(json.loads, f) if r["epoch"] == st["best_epoch"])
    res = {"run": run, "row": row_name, "gate_pos": row.get("gate_pos"), "seed": cfg["seed"], **meta,
           "epochs_run": st["epoch"], "early_stopped": st["patience"] >= cfg["patience"], "best_epoch": st["best_epoch"],
           "best_val_acc": st["best_acc"], **{k: v for k, v in best_rec.items() if k.startswith("val_") and k != "val_acc"}}
    if cfg["eval_test"]:
        res.update(test_metrics(model, test_loader, reg))
    res["gpu_hours"] = (st["elapsed"] + time.time() - t0) / 3600
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    os.remove(last_path)
    print(f"[{run}] done: best val acc {100 * st['best_acc']:.2f}% at epoch {st['best_epoch']} | "
          f"{res['gpu_hours']:.2f} GPU-h" + (f" | test {100 * res['test_acc']:.2f}%" if "test_acc" in res else ""))
    return res


# ---- results table ----------------------------------------------------------------------------
COLS = ["run", "pretrained_tag", "overhead_pct", "epochs_run", "best_epoch", "best_val_acc", "val_coverage",
        "val_frac_groups_erased", "val_median_group_contrast", "test_acc", "gpu_hours", "gpu", "timm", "code_sha"]


def summarize(out, dst=None):
    """Tabulate <out>/*/results.json; writes <dst>.csv and <dst>.md when dst is given."""
    res = []
    for p in sorted(glob.glob(os.path.join(out, "*", "results.json"))):
        with open(p) as f:
            res.append(json.load(f))
    cols = [c for c in COLS if any(c in r for r in res)]  # no test column unless test was opened

    def fmt(c, v):
        if v is None:
            return ""
        if c.endswith("_acc"):
            return f"{100 * v:.2f}"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |" for r in res]
    md.append(f"\nTotal GPU-hours: {sum(r['gpu_hours'] for r in res):.2f} over {len(res)} runs")
    text = "\n".join(md)
    if dst:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in res for k in r}))
            w.writeheader()
            w.writerows(res)
        with open(dst + ".md", "w") as f:
            f.write(text + "\n")
    print(text)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--row")
    ap.add_argument("--gate_pos", choices=["premerge", "postnorm"], help="required for row D")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="default: output/<config name>")
    ap.add_argument("--limit", type=int, help="smoke test: cap images per split")
    ap.add_argument("--epochs", type=int, help="override (smoke test)")
    ap.add_argument("--make_split", action="store_true", help="write the config's split file (once per dataset)")
    ap.add_argument("--summarize", metavar="DST", help="tabulate <out>/*/results.json into DST.csv and DST.md")
    args = ap.parse_args()
    if args.make_split:
        with open(args.config) as f:
            split = make_split(json.load(f))
        print(f"wrote {len(split['val'])} val images of {split['n_train_total']}")
    elif args.summarize:
        summarize(args.out or os.path.join("output", os.path.splitext(os.path.basename(args.config))[0]), args.summarize)
    else:
        train(args.config, args.row, args.gate_pos, args.seed, args.out, args.limit, args.epochs)


if __name__ == "__main__":
    main()
