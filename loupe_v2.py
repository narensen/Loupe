"""
loupe_v2.py -- The Loupe v2 for timm Swin (timm >= 0.9, NHWC stages; tested on timm 1.0.30).

Drop-in replacement for the v1 Loupe mask CNN. Insertion point unchanged: the gate acts on the
Stage-2 output (28x28 at 224px) at the Stage-2 -> Stage-3 boundary. Plug-and-play contract:

    model = build_loupe('swin_tiny_patch4_window7_224.ms_in1k', **ABLATION['D_both'])
    logits = model(images)                                          # same call as the backbone
    loss = F.cross_entropy(logits, y) + model.reg_loss('l1', lam=0.05)   # or 'coverage'

Flags (one module covers every ablation row):
  hidden    mask-head width h; default None = C2 // 4, the v1 rule (48 on Swin-T, 64 on Swin-B)
  mask_src  's2'   v1 head at 28x28: Conv3x3(C2->h) -> ReLU -> Conv1x1(h->1) -> sigmoid
            's1s2' hi-res head at 56x56: Conv3x3(C1->h)(F1) + up2(Conv1x1(C2->h)(F2)) -> ReLU
                   -> Conv1x1 -> sigmoid, then 2x2 avg-pool to 28x28 for gating (pooling preserves
                   mean coverage exactly, so the regularizer value is resolution-invariant)
            's1'   hi-res head from Stage-1 features only (Stage-1 semantics are weak; optional)
  head_attn None       plain head (v1)
            'cbam'     CBAM-style channel + spatial attention on the head's hidden features, after the
                       ReLU (reduction 4, 7x7 spatial conv; ~1.3k params on Swin-T). Built after the
                       head's other layers, so with and without CBAM the shared layers init identically
  gate      'mult'     g = M                      (v1)
            'residual' g = alpha + (1 - alpha) M  (alpha fixed, or learned inside alpha_bounds)
  gate_pos  'premerge' F2' = F2 * g, then Stage 3 unchanged (v1 placement)
            'postnorm' g, divided by its per-image spatial mean, scales each 28x28 sub-token AFTER
                       Stage-3's patch-merging LayerNorm (same boundary, same 28x28 gate, zero extra
                       parameters). A uniform gate is exactly identity, so init does not disturb the
                       pretrained network; with the residual gate, max/min gate ratio <= 1/alpha.

Why gate_pos exists: in timm, Stage 3 begins with PatchMerging = concat 2x2 group -> LayerNorm(4C)
-> Linear. LayerNorm is scale-invariant, so with 'premerge' any gate that is constant within a 2x2
group changes nothing (forward or backward) unless it is small enough for LN's eps to matter.
`python loupe_v2.py` measures this on your installed timm.
"""
from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models._manipulate import checkpoint_seq
except ImportError:  # older timm
    from timm.models.helpers import checkpoint_seq


def _nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W), channels_last strides, conv-friendly


class CBAM(nn.Module):
    """Channel then spatial attention (Woo et al., 2018) on the mask head's hidden features."""

    def __init__(self, c: int, reduction: int = 4, kernel: int = 7, padding_mode: str = "zeros"):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // reduction, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(c // reduction, c, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False, padding_mode=padding_mode)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h * torch.sigmoid(self.mlp(h.mean((2, 3), keepdim=True)) + self.mlp(h.amax((2, 3), keepdim=True)))
        return h * torch.sigmoid(self.spatial(torch.cat([h.mean(1, keepdim=True), h.amax(1, keepdim=True)], 1)))


class MaskHead(nn.Module):
    """Single-channel spatial mask. Returns M in [0,1] with shape (B,1,Hm,Wm)."""

    def __init__(self, src: str, c1: int, c2: int, hidden: int, padding_mode: str = "zeros", attn: str | None = None):
        super().__init__()
        self.src = src
        c_in = c2 if src == "s2" else c1
        self.conv3 = nn.Conv2d(c_in, hidden, 3, padding=1, padding_mode=padding_mode)
        self.lat = nn.Conv2d(c2, hidden, 1, bias=False) if src == "s1s2" else None
        self.out = nn.Conv2d(hidden, 1, 1)
        assert attn in (None, "cbam"), attn
        self.attn = CBAM(hidden, padding_mode=padding_mode) if attn == "cbam" else None  # built last: see docstring

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        h = self.conv3(_nchw(f2 if self.src == "s2" else f1))
        if self.lat is not None:  # Stage-2 semantics, projected then upsampled to Stage-1 grid
            h = h + F.interpolate(self.lat(_nchw(f2)), size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = F.relu(h)
        if self.attn is not None:
            h = self.attn(h)
        return torch.sigmoid(self.out(h))


def gated_patch_merging(pm: nn.Module, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """timm PatchMerging with a per-sub-token gate applied after its LayerNorm.
    x: (B,H,W,C), g: (B,H,W,1). The gate is reshaped exactly like x, so the 4C layout matches."""
    B, H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "postnorm gate needs even H and W"
    x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    gg = g.reshape(B, H // 2, 2, W // 2, 2, 1).permute(0, 1, 3, 4, 2, 5)
    x = pm.norm(x)
    x = (x.unflatten(-1, (2, 2, C)) * gg).flatten(3)
    return pm.reduction(x)


def probe_transform(m: torch.Tensor, mode: str = "none", alpha: float = 0.3) -> torch.Tensor:
    """Inference-time interventions on a trained mask m: (B,1,H,W), H and W even (permute first if
    your v1 mask is (B,H,W,1)). Paste into v1 right before F * M to see which mechanism v1 uses.
      'no_erasure'  divide each 2x2 group by its max: under premerge this is identical to v1
                    except where LN's eps mattered, so any accuracy change = erasure's share
      'no_contrast' replace each 2x2 group by its mean: removes within-group contrast
      'gate_off'    M = 1 everywhere
      'floor'       alpha + (1 - alpha) M, the residual gate applied post hoc"""
    if mode == "none":
        return m
    if mode == "gate_off":
        return torch.ones_like(m)
    if mode == "floor":
        return alpha + (1 - alpha) * m
    mf = m.float()
    if mode == "no_erasure":
        mx = F.interpolate(F.max_pool2d(mf, 2), scale_factor=2, mode="nearest")
        out = torch.where(mx > 0, mf / mx.clamp_min(1e-30), torch.ones_like(mf))  # fp16 underflow -> uniform
    elif mode == "no_contrast":
        out = F.interpolate(F.avg_pool2d(mf, 2), scale_factor=2, mode="nearest")
    else:
        raise ValueError(mode)
    return out.to(m.dtype)


class LoupeSwin(nn.Module):
    def __init__(self, backbone: nn.Module, mask_src: str = "s2", gate: str = "mult",
                 gate_pos: str = "premerge", alpha: float = 0.3, learn_alpha: bool = False,
                 alpha_bounds=(0.2, 0.5), hidden: int | None = None, padding_mode: str = "zeros",
                 head_attn: str | None = None):
        super().__init__()
        assert mask_src in ("s2", "s1", "s1s2"), mask_src
        assert gate in ("mult", "residual"), gate
        assert gate_pos in ("premerge", "postnorm"), gate_pos
        self.backbone = backbone
        self.mask_src, self.gate, self.gate_pos = mask_src, gate, gate_pos
        c1, c2 = backbone.feature_info[0]["num_chs"], backbone.feature_info[1]["num_chs"]
        # v1 width rule; conv3 -> (lat) -> out construction order matches v1's RNG use for row A
        self.head = MaskHead(mask_src, c1, c2, hidden or c2 // 4, padding_mode, head_attn)
        self.alpha_bounds = tuple(float(a) for a in alpha_bounds)
        self.alpha_fixed = float(alpha)
        if gate == "residual" and learn_alpha:
            lo, hi = self.alpha_bounds
            assert lo < alpha < hi, "init alpha must lie strictly inside alpha_bounds"
            p = (alpha - lo) / (hi - lo)
            self.alpha_logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        else:
            self.register_parameter("alpha_logit", None)
        self.probe_mode = "none"  # set to a probe_transform mode for inference-time probes
        self._m_hi = self._m = self._g = None
        self._check_contract()

    # ---- contract -------------------------------------------------------------------------
    def _check_contract(self):
        bb = self.backbone
        ok = hasattr(bb, "layers") and len(bb.layers) >= 3 and hasattr(bb.layers[2], "downsample")
        pm = bb.layers[2].downsample if ok else None
        assert ok and hasattr(pm, "norm") and hasattr(pm, "reduction"), \
            "expects timm >= 0.9 SwinTransformer (Stage 3 must start with PatchMerging)"
        w = pm.norm.weight
        with torch.no_grad():
            x = torch.randn(2, 4, 4, w.numel() // 4, device=w.device, dtype=w.dtype)
            tol = 1e-5 if w.dtype == torch.float32 else 1e-2
            same = torch.allclose(gated_patch_merging(pm, x, torch.ones_like(x[..., :1])), pm(x), atol=tol)
        assert same, "PatchMerging layout differs in this timm version; postnorm gate would be misaligned"

    def alpha(self):
        if self.alpha_logit is None:
            return self.alpha_fixed
        lo, hi = self.alpha_bounds
        return lo + (hi - lo) * torch.sigmoid(self.alpha_logit)

    # ---- forward --------------------------------------------------------------------------
    def stage12(self, x):
        bb = self.backbone
        f1 = bb.layers[0](bb.patch_embed(x))  # (B, H/4, W/4, C1)   56x56 at 224
        f2 = bb.layers[1](f1)                 # (B, H/8, W/8, C2)   28x28 at 224 (v1 insertion point)
        return f1, f2

    def from_stage2(self, f2, g):
        """g: (B, H/8, W/8, 1). Runs Stage 3 (gated as configured), Stage 4, norm, head."""
        bb = self.backbone
        st3 = bb.layers[2]
        if self.gate_pos == "premerge":
            x = st3(f2 * g)
        else:
            g = g / g.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)  # unit spatial mean: uniform gate == identity
            x = gated_patch_merging(st3.downsample, f2, g)
            x = checkpoint_seq(st3.blocks, x) if getattr(st3, "grad_checkpointing", False) else st3.blocks(x)
        for st in bb.layers[3:]:
            x = st(x)
        return bb.forward_head(bb.norm(x))

    def forward(self, x):
        f1, f2 = self.stage12(x)
        m_hi = self.head(f1, f2)                                    # (B,1,Hm,Wm)
        m = m_hi if m_hi.shape[-2:] == f2.shape[1:3] else F.adaptive_avg_pool2d(m_hi, f2.shape[1:3])
        m = probe_transform(m, self.probe_mode)                     # no-op unless probing
        a = self.alpha()
        g = m if self.gate == "mult" else a + (1 - a) * m          # (B,1,28,28)
        g = g.permute(0, 2, 3, 1)                                   # (B,28,28,1) NHWC
        self._m_hi, self._m, self._g = m_hi, m, g
        return self.from_stage2(f2, g)

    # ---- loss hook ------------------------------------------------------------------------
    def reg_loss(self, kind: str = "l1", lam: float = 0.05, tau=(0.05, 0.15),
                 lam_cov: float = 1.0, lam_l1: float = 0.0) -> torch.Tensor:
        """Regularizer on the mask from the latest forward, in fp32.
        'l1'       lam * mean(M), per-element mean over (B,1,H,W): identical to v1's torch.mean(attention_map).
        'coverage' lam_cov * mean_b[relu(lo - c_b)^2 + relu(c_b - hi)^2] + lam_l1 * mean(M),
                   c_b = per-image mean of M. tau=(t, t) gives the two-sided (c_b - t)^2 target."""
        cov = self._m_hi.float().mean(dim=(1, 2, 3))
        if kind == "l1":
            return lam * cov.mean()
        if kind == "coverage":
            lo, hi = tau
            pen = F.relu(lo - cov).pow(2) + F.relu(cov - hi).pow(2)
            return lam_cov * pen.mean() + lam_l1 * cov.mean()
        raise ValueError(kind)

    @torch.no_grad()
    def gate_stats(self) -> dict:
        """Cheap per-batch diagnostics for logging (latest forward)."""
        m = self._m.float()
        g = self._g.float().permute(0, 3, 1, 2)
        gmax, gmin = F.max_pool2d(g, 2), -F.max_pool2d(-g, 2)
        s = {"coverage": m.mean().item(),
             "frac_m_lt_1e-3": (m < 1e-3).float().mean().item(),
             "frac_groups_erased": (gmax < 1e-3).float().mean().item(),
             "median_group_contrast": (gmax / gmin.clamp_min(1e-6)).median().item()}
        if self.gate == "residual":
            s["alpha"] = float(self.alpha())
        return s

    def mask_for_eval(self, size):
        """Mask at native resolution, upsampled to image size, for part-IoU evaluation (no training use)."""
        return F.interpolate(self._m_hi.float(), size=size, mode="bilinear", align_corners=False)


# ---- builders / configs -----------------------------------------------------------------------
ABLATION = {
    "A_v1":       dict(mask_src="s2",   gate="mult",     gate_pos="premerge"),
    "B_residual": dict(mask_src="s2",   gate="residual", gate_pos="premerge", alpha=0.3),
    "C_hires":    dict(mask_src="s1s2", gate="mult",     gate_pos="premerge"),
    "D_both":     dict(mask_src="s1s2", gate="residual", gate_pos="premerge", alpha=0.3),
}
ABLATION_POSTNORM = {  # LN-aware placement; run per the probe / screen plan
    "A2_v1_postnorm":       dict(ABLATION["A_v1"], gate_pos="postnorm"),
    "B2_residual_postnorm": dict(ABLATION["B_residual"], gate_pos="postnorm"),
    "D2_both_postnorm":     dict(ABLATION["D_both"], gate_pos="postnorm"),
}


def build_loupe(model_name: str = "swin_tiny_patch4_window7_224", num_classes: int = 200,
                pretrained: bool = True, **loupe_kwargs) -> LoupeSwin:
    import timm
    # create -> reset_classifier -> mask head: same RNG order and default nn.Linear init as v1's
    # SwinWithLoupe (create_model(..., num_classes=200) would trunc_normal-init the head instead)
    bb = timm.create_model(model_name, pretrained=pretrained)
    bb.reset_classifier(num_classes)
    return LoupeSwin(bb, **loupe_kwargs)


def load_v1_state_dict(model: LoupeSwin, sd: dict):
    """Load a v1 SwinWithLoupe checkpoint into a row-A LoupeSwin (mask_src='s2', gate='mult')."""
    ren = {"loupe.attention_net.0.": "head.conv3.", "loupe.attention_net.2.": "head.out.", "head.": "backbone.head.fc."}
    out = {}
    for k, v in sd.items():
        new = next((n + k[len(o):] for o, n in ren.items() if k.startswith(o)), "backbone." + k)
        out[new] = v
    return model.load_state_dict(out)


def param_groups(model: nn.Module, lr: float, weight_decay: float, alpha_lr: float = 1e-3):
    """v1 used one lr for everything. Lion steps are lr*sign(.), so at lr=1e-5 a learned alpha logit
    moves at most ~0.2 in a full run (alpha barely changes). Give alpha its own lr if you learn it."""
    alpha = [p for n, p in model.named_parameters() if n.endswith("alpha_logit")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("alpha_logit")]
    groups = [{"params": rest, "lr": lr, "weight_decay": weight_decay}]
    if alpha:
        groups.append({"params": alpha, "lr": alpha_lr, "weight_decay": 0.0})
    return groups


# ---- self-test ---------------------------------------------------------------------------------
@torch.no_grad()
def ln_check(model: LoupeSwin, x: torch.Tensor):
    """How much do the logits move when the Stage-2 map is gated, by placement?"""
    model.eval()
    _, f2 = model.stage12(x)
    B, H, W, _ = f2.shape
    ones = f2.new_ones(B, H, W, 1)
    pos0 = model.gate_pos
    rows = []
    gens = {f"uniform {c:g}": (lambda c=c: ones * c) for c in (0.3, 0.05, 1e-2, 1e-3, 1e-4)}
    gen = torch.Generator(device=f2.device).manual_seed(0)
    rand = lambda *shape: torch.rand(*shape, generator=gen, device=f2.device, dtype=f2.dtype)
    blk = rand(B, 1, H // 2, W // 2) * 0.7 + 0.3                # constant within each 2x2 group, in [0.3, 1]
    gens["2x2-constant in [0.3,1]"] = lambda: F.interpolate(blk, scale_factor=2, mode="nearest").permute(0, 2, 3, 1)
    per_tok = rand(B, H, W, 1) * 0.7 + 0.3                       # varies inside groups too
    gens["per-token in [0.3,1]"] = lambda: per_tok
    for pos in ("premerge", "postnorm"):
        model.gate_pos = pos
        ref = model.from_stage2(f2, ones)
        for name, gen in gens.items():
            z = model.from_stage2(f2, gen())
            rel = ((z - ref).norm() / ref.norm()).item()
            agree = (z.argmax(1) == ref.argmax(1)).float().mean().item()
            rows.append((pos, name, rel, agree))
    model.gate_pos = pos0
    return rows


def selftest(model_name: str, pretrained: bool, hidden: int, images: str = ""):
    import glob
    import timm
    torch.manual_seed(0)
    # with real images keep the pretrained classifier so top-1 agreement is meaningful
    bb = timm.create_model(model_name, pretrained=pretrained, **({} if images else {"num_classes": 200})).eval()
    if images:  # real photos + pretrained weights give realistic feature scales for the LN check
        from PIL import Image
        cfg = timm.data.resolve_data_config({}, model=bb)
        cfg.update(crop_pct=224 / 256)  # resize 256 -> center crop 224, as in v1 testing
        tf = timm.data.create_transform(**cfg)
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in sorted(glob.glob(images))])
    else:
        x = torch.randn(2, 3, 224, 224)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    n_bb = sum(p.numel() for p in bb.parameters())
    with torch.no_grad():
        ref = bb(x)
    print(f"timm {timm.__version__} | {model_name} | pretrained={pretrained} | backbone {n_bb/1e6:.2f}M | hidden={hidden}")

    for name, cfg in {**ABLATION, **ABLATION_POSTNORM}.items():
        torch.manual_seed(0)
        model = LoupeSwin(bb, hidden=hidden, **cfg)  # wraps the same backbone; nothing in it is modified
        n_add = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("backbone."))
        model.eval()
        with torch.no_grad():
            model.probe_mode = "gate_off"
            same = torch.allclose(model(x), ref, atol=1e-4)
            model.probe_mode = "none"
            z0 = model(x)
        init_shift = ((z0 - ref).norm() / ref.norm()).item()
        model.train()
        bb.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y) + model.reg_loss("coverage") + model.reg_loss("l1")
        loss.backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
        print(f"{name:21s} +{n_add:6d} ({100*n_add/n_bb:.3f}%) | gate_off==backbone {same} | grads {grads_ok} | "
              f"init logit shift {init_shift:.3f} | M_hi {tuple(model._m_hi.shape[1:])} -> M {tuple(model._m.shape[1:])}")
    bb.zero_grad(set_to_none=True)
    with torch.no_grad():
        f1, f2 = model.stage12(x)
    print(f"shapes: x {tuple(x.shape)} -> F1 {tuple(f1.shape)} -> F2 {tuple(f2.shape)} (gated) -> logits {tuple(ref.shape)}")

    model = LoupeSwin(bb, hidden=hidden, learn_alpha=True, **ABLATION["D_both"]).train()
    (F.cross_entropy(model(x), y) + model.reg_loss("coverage")).backward()
    print(f"learned alpha={float(model.alpha().detach()):.3f}, grad={model.alpha_logit.grad.item():+.2e}")
    bb.zero_grad(set_to_none=True)

    pm = copy.deepcopy(bb.layers[2].downsample)
    pm.norm = nn.Identity()  # without LN, postnorm gating must equal premerge gating exactly
    f = torch.randn(2, 28, 28, bb.feature_info[1]["num_chs"])
    g = torch.rand(2, 28, 28, 1)
    print("postnorm layout == premerge when LN removed:", torch.allclose(gated_patch_merging(pm, f, g), pm(f * g), atol=1e-5))

    print(f"Stage-2 output per-element std: {f2.std().item():.3f}")
    print("LN check: relative logit change vs ungated, top-1 agreement")
    for pos, name, rel, agree in ln_check(LoupeSwin(bb, hidden=hidden), x):
        print(f"  {pos:9s} {name:26s} rel={rel:.2e} top1_agree={agree:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swin_tiny_patch4_window7_224")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--hidden", type=int, default=None, help="default C2//4 (v1 rule)")
    ap.add_argument("--images", default="", help="optional glob of real images, e.g. 'cub_val/*.jpg'")
    a = ap.parse_args()
    selftest(a.model, a.pretrained, a.hidden, a.images)
